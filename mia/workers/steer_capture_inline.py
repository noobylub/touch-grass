"""Inline vector admission for the served steer_capture worker only."""
from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from pathlib import Path
import tempfile

import torch


def resolve_inline_steering(engine, config, check_capacity):
    """Validate a direction and resolve it to an immutable worker-readable file."""
    if "vector_path" in config:
        raise ValueError("provide steer.vector or steer.vector_path, not both")
    model_config = getattr(engine, "model_config", None)
    if model_config is None:
        model_config = engine.vllm_config.model_config
    text_config = model_config.hf_text_config
    width = int(text_config.hidden_size)
    values = config["vector"]
    if (not isinstance(values, list) or len(values) != width
            or any(type(value) not in (int, float) for value in values)):
        raise ValueError(f"steer.vector must be a numeric list of shape [{width}]")
    direction = torch.tensor(values, dtype=torch.float32)
    dtype = model_config.dtype
    if not torch.isfinite(direction).all() or not torch.isfinite(direction.to(dtype)).all():
        raise ValueError("steer.vector must remain finite in the model dtype")
    method = config.get("method", "adjust_rs")
    if method not in ("add_vector", "adjust_rs"):
        raise ValueError("steer.method must be add_vector or adjust_rs")
    coefficient = config.get("coefficient", 0)
    if type(coefficient) not in (int, float) or not math.isfinite(coefficient):
        raise ValueError("steer.coefficient must be finite")
    selection = config.get("optimal_layer", -1)
    if selection != "all":
        layers = selection if isinstance(selection, list) else [selection]
        if not layers or any(type(layer) is not int
                             or not 0 <= layer < int(text_config.num_hidden_layers)
                             for layer in layers):
            raise ValueError("steer.optimal_layer contains an invalid layer index")
    if config.get("phase", "both") not in ("prefill", "decode", "both"):
        raise ValueError("invalid steer.phase")
    if config.get("positions", "all_tokens") not in ("all_tokens", "last_token"):
        raise ValueError("invalid steer.positions")
    payload = {"dir": direction}
    if method == "adjust_rs":
        projection = config.get("avg_proj")
        if (type(projection) not in (int, float) or not math.isfinite(projection)
                or not torch.isfinite(torch.tensor(projection, dtype=dtype))):
            raise ValueError("adjust_rs requires a finite avg_proj in the model dtype")
        if not torch.isclose(direction.norm(), torch.tensor(1.0), atol=1e-3, rtol=1e-3):
            raise ValueError("adjust_rs requires a unit-length vector")
        payload["avg_proj"] = torch.tensor(projection, dtype=torch.float32)
    digest = hashlib.sha256(direction.numpy().tobytes())
    if "avg_proj" in payload:
        digest.update(payload["avg_proj"].numpy().tobytes())
    storage = getattr(engine, "_mia_inline_vector_storage", None)
    if storage is None:
        storage = tempfile.TemporaryDirectory(prefix="mia-steer-capture-")
        engine._mia_inline_vector_storage = storage
    path = Path(storage.name) / (digest.hexdigest() + ".pt")
    check_capacity(engine, str(path))
    if not path.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=storage.name, delete=False) as handle:
                temporary = Path(handle.name)
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    resolved = {key: value for key, value in config.items()
                if key not in ("vector", "avg_proj")}
    resolved["vector_path"] = str(path)
    return resolved


def _inline_path(engine, path: str) -> Path | None:
    storage = getattr(engine, "_mia_inline_vector_storage", None)
    candidate = Path(path)
    if storage is not None and candidate.parent == Path(storage.name):
        return candidate
    return None


def is_inline_vector_path(engine, path: str) -> bool:
    """Whether ``path`` belongs to this engine's generated inline files."""
    return _inline_path(engine, path) is not None


async def pin_steering_vector(engine, path: str) -> None:
    """Pin a path for one generation, evicting the oldest idle combined-worker entry."""
    import asyncio

    lock = getattr(engine, "_mia_steer_fifo_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        engine._mia_steer_fifo_lock = lock
    async with lock:
        entries = getattr(engine, "_mia_steer_fifo_entries", None)
        if entries is None:
            entries = OrderedDict()
            engine._mia_steer_fifo_entries = entries
        if path in entries:
            entries[path] += 1
            return

        capacity = int(os.environ.get("MIA_STEER_VMAX", "16"))
        if len(entries) >= capacity:
            victim = next((key for key, references in entries.items()
                           if references == 0), None)
            if victim is None:
                inline_file = _inline_path(engine, path)
                if inline_file is not None:
                    inline_file.unlink(missing_ok=True)
                raise RuntimeError(
                    "all steering vector slots are in use; retry when a generation finishes")
            await engine.collective_rpc("evict_steering_vector", args=(victim,))
            entries.pop(victim)
            getattr(engine, "_mia_steer_interned", {}).pop(victim, None)
            getattr(engine, "_mia_steer_loaded", {}).pop(victim, None)
            victim_file = _inline_path(engine, victim)
            if victim_file is not None:
                victim_file.unlink(missing_ok=True)

        entries[path] = 1
        getattr(engine, "_mia_steer_interned", {})[path] = True


async def unpin_steering_vector(engine, path: str) -> None:
    """Release one request's reference; keep the cache entry in FIFO order."""
    lock = getattr(engine, "_mia_steer_fifo_lock", None)
    entries = getattr(engine, "_mia_steer_fifo_entries", {})
    if lock is None:
        return
    async with lock:
        references = entries.get(path)
        if references is not None:
            entries[path] = max(0, references - 1)