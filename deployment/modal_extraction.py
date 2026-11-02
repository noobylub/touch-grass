# Extract hidden-state directions, save them to the shared Volume, and return vector IDs.
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import secrets
import struct
import tempfile
import threading
import uuid
from pathlib import Path

import httpx
import modal
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.environ.get("MIA_MODEL")
LAYERS = os.environ.get("MIA_LAYERS", "10,21")
MAX_IN_FLIGHT = int(os.environ.get("MIA_MAX_IN_FLIGHT", "32"))
ACCUMULATION_ROUTE = "/v1/mia/hidden-state-accumulation"
VECTOR_DIR = "/vectors/steering-vectors"
VECTOR_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_VECTOR_STORE_LOCK = threading.Lock()


def resolve_vector_name(payload: dict) -> str | None:
    """Validate and normalize an optional extraction vector name."""
    if "vector_name" not in payload:
        return None
    vector_name = payload["vector_name"]
    if not isinstance(vector_name, str) or not vector_name.strip():
        raise ValueError("vector_name must be a non-empty string")
    vector_name = vector_name.strip()
    if len(vector_name) > 120:
        raise ValueError("vector_name must be at most 120 characters")
    return vector_name


def unique_vector_name(vector_name: str, layer_id: int, existing_names: set[str]) -> str:
    """Add a layer label and a random suffix when the name is already stored."""
    base_name = f"{vector_name}-layer-{layer_id}"
    if base_name not in existing_names:
        return base_name
    while True:
        candidate = f"{base_name}-{secrets.token_hex(4)}"
        if candidate not in existing_names:
            return candidate


def resolve_requested_layers(payload: dict, default_layers: str) -> list[int] | str:
    """Use a valid request layer list, or fall back to the configured default."""
    if "layers" not in payload:
        return default_layers

    requested_layers = payload["layers"]
    if (
        not isinstance(requested_layers, list)
        or not requested_layers
        or any(
            not isinstance(layer, int) or isinstance(layer, bool)
            for layer in requested_layers
        )
    ):
        raise ValueError("layers must be a non-empty list of integer layer indexes")

    return requested_layers


app = modal.App("ibm-vllm-hook")

image = (
    # Initialising an Image 
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ninja-build",
        "build-essential",
    )
    .add_local_dir(
        ".",
        "/opt/vLLM-Hook",
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".env",
            "__pycache__",
            "*.pt",
            "*.safetensors",
        ],
    )
    .run_commands(
        # Verify CUDA compiler is available
        "which nvcc && nvcc --version",

        # Install IBM's pinned dependencies
        "cd /opt/vLLM-Hook && pip install -r requirement.txt",

        "pip uninstall -y torchcodec",

        # Install your locally cloned IBM Hook
        "cd /opt/vLLM-Hook && pip install -e . --no-deps",

        # Verify installed PyTorch CUDA version
        "python -c 'import torch; print(\"PyTorch CUDA:\", torch.version.cuda)'",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "MIA_MODEL": MODEL,
        "MIA_WORKER": "hidden_state_accum",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HOME": "/cache/huggingface",
    })
)


cache = modal.Volume.from_name(
    "ibm-model-cache",
    create_if_missing=True,
)
vector_store = modal.Volume.from_name(
    "steering-vector-store",
    create_if_missing=True,
)


def store_extracted_vector(
    direction, layer_id: int, vector_name: str | None = None
) -> tuple[str, bool, str | None]:
    """Store a model- and layer-specific float32 direction and return its ID."""
    import torch

    vector = np.asarray(direction, dtype="<f4")
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("extracted steering vector must be a finite 1D vector")
    digest = hashlib.sha256()
    digest.update(b"mia-steering-vector-v2\0")
    digest.update(MODEL.encode("utf-8"))
    digest.update(int(layer_id).to_bytes(8, "little"))
    digest.update(int(vector.size).to_bytes(8, "little"))
    digest.update(vector.tobytes())
    with _VECTOR_STORE_LOCK:
        stored_name = None
        if vector_name is not None:
            existing_names = set()
            if os.path.isdir(VECTOR_DIR):
                for entry in os.scandir(VECTOR_DIR):
                    if not entry.is_file() or not entry.name.endswith(".pt"):
                        continue
                    try:
                        saved = torch.load(entry.path, map_location="cpu", weights_only=True)
                    except (OSError, RuntimeError, ValueError, EOFError):
                        continue
                    if isinstance(saved, dict) and isinstance(saved.get("name"), str):
                        existing_names.add(saved["name"])
            stored_name = unique_vector_name(vector_name, layer_id, existing_names)
            digest.update(b"\0name\0")
            digest.update(stored_name.encode("utf-8"))
        vector_hash = digest.hexdigest()
        path = os.path.join(VECTOR_DIR, f"{vector_hash}.pt")
        if os.path.isfile(path):
            return vector_hash, False, stored_name
        os.makedirs(VECTOR_DIR, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=VECTOR_DIR, prefix="vector-", delete=False
            ) as temporary:
                temporary_path = temporary.name
            torch.save({
                "dir": torch.from_numpy(vector.copy()),
                "layer_id": int(layer_id),
                "name": stored_name,
            }, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
    return vector_hash, True, stored_name

def fetch_result(base_url: str, api_key: str, delivery_id: str, timeout_s: float = 90.0) -> dict:
    """Fetch a group result directly, without importing the server-side MIA package."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    response = httpx.get(
        root + ACCUMULATION_ROUTE,
        params={"delivery_id": delivery_id},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )

    # Prechecks: include the server detail when the result endpoint rejects a request.
    if not response.is_success:
        detail = response.text[:1000]
        raise RuntimeError(
            f"accumulation result request failed ({response.status_code}): {detail}"
        )
    data = response.content
    if len(data) < 8:
        raise ValueError("accumulation response is truncated")
    header_size = struct.unpack("<Q", data[:8])[0]
    if header_size > len(data) - 8:
        raise ValueError("accumulation response metadata is truncated")
    try:
        header = json.loads(data[8:8 + header_size].decode("utf-8"))
        metadata = header.get("__metadata__") or {}
        layer_ids = json.loads(metadata["layers"])
        layer_names_metadata = json.loads(metadata["layer_names"])
    except (AttributeError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid accumulation response metadata") from exc
    if metadata.get("format") != "mia-hidden-state-accum-v1":
        raise ValueError("unsupported accumulation response format")
    if not isinstance(layer_ids, list) or not isinstance(layer_names_metadata, dict):
        raise ValueError("invalid accumulation layer metadata")

    # Decode section: read the already-validated tensors and build layer results.
    buffer_start = 8 + header_size
    dtype_map = {"F32": np.dtype("<f4"), "I64": np.dtype("<i8")}

    def read_tensor(name: str) -> np.ndarray:
        spec = header[name]
        dtype = dtype_map.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported tensor dtype: {spec['dtype']}")
        start, end = spec["data_offsets"]
        return np.frombuffer(data, dtype=dtype, count=(end - start) // dtype.itemsize,
                offset=buffer_start + start).reshape(spec["shape"]).copy()

    layers = {}
    for layer in layer_ids:
        layer = int(layer)
        layers[layer] = {
            "sum": read_tensor(f"{layer}/sum"),
            "mean": read_tensor(f"{layer}/mean"),
            "count": int(read_tensor(f"{layer}/count").item()),
        }
    layer_names = {
        int(layer): name
        for layer, name in layer_names_metadata.items()
    }
    return {"layers": layers, "layer_names": layer_names}





@app.server(
    image=image,
    gpu="L40S",
    port=8770,
    volumes={"/cache": cache, "/vectors": vector_store},
    secrets=[modal.Secret.from_name("vllm-auth")],
    startup_timeout=1800,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    unauthenticated=True,
)
class VLLMServer:

    @modal.enter()
    def start(self):
        import subprocess
        import threading
        import time

        import uvicorn
        from fastapi import FastAPI, Header, HTTPException

        api_key = os.environ["VLLM_API_KEY"]
        internal_url = "http://127.0.0.1:8771"
        base_url = f"{internal_url}/v1"

        self.process = subprocess.Popen([
            "vllm", "serve", MODEL,
            "--host", "127.0.0.1",
            "--port", "8771",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.7",
            "--no-enable-prefix-caching",
            "--api-key", api_key,
        ], cwd="/opt/vLLM-Hook")

        # Wait for model startup before exposing the wrapper.
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("vLLM exited during startup")
            try:
                if httpx.get(f"{internal_url}/health", timeout=2).is_success:
                    break
            except httpx.RequestError:
                pass
            time.sleep(2)
        else:
            self.process.terminate()
            raise RuntimeError("vLLM startup timed out")
        print("--------------------"*20)
        print(f"[startup] vLLM is ready. Model: {MODEL}", flush=True)

        # --------
        # MODAL ROUTES
        # --------
        # Modal routes traffic to port 8770; vLLM stays private on port 8771.
        self.web_app = FastAPI()

        @self.web_app.get("/health")
        async def health():
            return {"status": "ok"}

        @self.web_app.get("/vectors")
        async def list_vectors(authorization: str = Header("")):
            if not secrets.compare_digest(authorization, f"Bearer {api_key}"):
                raise HTTPException(status_code=401, detail="Invalid API key")

            import torch

            await vector_store.reload.aio()
            with _VECTOR_STORE_LOCK:
                vector_files = sorted(
                    (
                        path for path in os.scandir(VECTOR_DIR)
                        if path.is_file() and path.name.endswith(".pt")
                        and len(path.name) == 67
                    ),
                    key=lambda entry: entry.name,
                ) if os.path.isdir(VECTOR_DIR) else []
                vectors = []
                for entry in vector_files:
                    vector_hash = entry.name[:-3]
                    if any(char not in "0123456789abcdef" for char in vector_hash):
                        continue
                    try:
                        saved = torch.load(entry.path, map_location="cpu", weights_only=True)
                        direction = saved.get("dir") if isinstance(saved, dict) else saved
                        if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
                            continue
                    except (OSError, RuntimeError, ValueError, EOFError):
                        continue
                    stat = entry.stat()
                    vectors.append({
                        "vector_id": vector_hash,
                        "layer_id": saved.get("layer_id") if isinstance(saved, dict) else None,
                        "name": saved.get("name") if isinstance(saved, dict) else None,
                        "model": MODEL,
                        "hidden_size": int(direction.numel()),
                        "dtype": str(direction.dtype),
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                    })
            return {"model": MODEL, "count": len(vectors), "vectors": vectors}

        @self.web_app.delete("/vectors/{vector_id}")
        async def delete_vector(vector_id: str, authorization: str = Header("")):
            if not secrets.compare_digest(authorization, f"Bearer {api_key}"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            if VECTOR_ID_RE.fullmatch(vector_id) is None:
                raise HTTPException(
                    status_code=422,
                    detail="vector_id must be a 64-character lowercase SHA-256 hash",
                )

            await vector_store.reload.aio()
            vector_path = os.path.join(
                VECTOR_DIR, f"{vector_id}.pt"
            )
            with _VECTOR_STORE_LOCK:
                if not os.path.isfile(vector_path):
                    raise HTTPException(status_code=404, detail="vector_id was not found")
                os.remove(vector_path)
            await vector_store.commit.aio()
            return {"deleted": True, "vector_id": vector_id}

        # Send the prompts to vLLM, get their hidden-state means, subtract the
        # negative means from the positive means, and save the resulting vectors.
        @self.web_app.post("/extract-vectors")
        async def extract_vectors(payload: dict) -> dict[str, dict]:
            try:
                requested_layers = resolve_requested_layers(payload, LAYERS)
                requested_vector_name = resolve_vector_name(payload)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=str(exc),
                ) from exc
            pairs = payload.get("pairs")
            if not isinstance(pairs, list) or not pairs:
                raise HTTPException(status_code=422, detail="pairs must be a non-empty list")
            prompts_by_group: dict[str, list[str]] = {}
            pair_groups: dict[str, dict[str, str]] = {}
            for pair in pairs:
                if not isinstance(pair, dict):
                    raise HTTPException(status_code=422, detail="each pair must be an object")
                pair_id = pair.get("pair_id")
                if not isinstance(pair_id, str) or not pair_id.strip() or pair_id in pair_groups:
                    raise HTTPException(status_code=422, detail="pair_id must be non-empty and unique")
                pair_groups[pair_id] = {}
                for role in ("positive", "negative"):
                    prompts = pair.get(role)
                    if not isinstance(prompts, list) or not prompts or any(
                        not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
                    ):
                        raise HTTPException(
                            status_code=422,
                            detail=f"{pair_id}: {role} must be a non-empty list of non-empty prompts",
                        )
                    group_id = uuid.uuid4().hex
                    pair_groups[pair_id][role] = group_id
                    prompts_by_group[group_id] = prompts

                    # Each positive or negative prompt set gets its own MIA group and result ID.
            delivery_ids = {group_id: uuid.uuid4().hex for group_id in prompts_by_group}
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)

            # Keep each prompt's capture metadata isolated while allowing vLLM
            # to batch the concurrent requests internally.
            async def submit(group_id: str, prompt: str) -> None:
                extra_args = {
                    "learning_enable": True,
                    "learning_group_id": group_id,
                    "learning_group_size": len(prompts_by_group[group_id]),
                    "learning_delivery_id": delivery_ids[group_id],
                    "learning_layers": requested_layers,
                    "learning_phase": "prefill",
                    "learning_positions": "all_tokens",
                }
                await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=0,
                    extra_body={"vllm_xargs": extra_args},
                )

            requests = [
                (group_id, prompt)
                for group_id, prompts in prompts_by_group.items()
                for prompt in prompts
            ]
            try:
                # Submit each prompt as a separate request. Chunks only limit
                # concurrency; the group metadata keeps MIA results separate.
                for start in range(0, len(requests), MAX_IN_FLIGHT):
                    chunk = requests[start:start + MAX_IN_FLIGHT]
                    outcomes = await asyncio.gather(
                        *(submit(group_id, prompt) for group_id, prompt in chunk),
                        return_exceptions=True,
                    )
                    for outcome in outcomes:
                        if isinstance(outcome, BaseException):
                            raise outcome
            finally:
                await client.close()

            # MIA publishes each group only after its declared members finish.
            group_results = {}
            for group_id in prompts_by_group:
                group_results[group_id] = await asyncio.to_thread(
                    fetch_result, base_url, api_key, delivery_ids[group_id]
                )
            results = {}
            vector_store_changed = False
            for pair_id, groups in pair_groups.items():
                positive = group_results[groups["positive"]]
                negative = group_results[groups["negative"]]
                if positive["layers"].keys() != negative["layers"].keys():
                    raise RuntimeError(f"{pair_id}: positive and negative layers do not match")
                layers = {}
                # Subtract the completed positive/negative means to create each layer direction.
                for layer_id, positive_values in positive["layers"].items():
                    negative_values = negative["layers"][layer_id]
                    positive_mean = positive_values["mean"]
                    negative_mean = negative_values["mean"]
                    if positive_mean.shape != negative_mean.shape:
                        raise RuntimeError(f"{pair_id}: mean shapes do not match at layer {layer_id}")
                    direction = positive_mean - negative_mean
                    # Storing vector here
                    vector_id, created, stored_name = store_extracted_vector(
                        direction, layer_id, requested_vector_name
                    )
                    vector_store_changed = vector_store_changed or created
                    layers[layer_id] = {
                        "dir": direction.tolist(),
                        "vector_id": vector_id,
                        "name": stored_name,
                        "positive_mean": positive_mean.tolist(),
                        "negative_mean": negative_mean.tolist(),
                        "positive_count": positive_values["count"],
                        "negative_count": negative_values["count"],
                    }
                results[pair_id] = {"model": MODEL, "layer_names": positive["layer_names"], "layers": layers}
            if vector_store_changed:
                with _VECTOR_STORE_LOCK:
                    await vector_store.commit.aio()
            return results

        self.web_server = uvicorn.Server(
            uvicorn.Config(
                self.web_app,
                host="0.0.0.0",
                port=8770,
                log_level="info",
            )
        )
        self.web_thread = threading.Thread(
            target=self.web_server.run,
            name="extraction-api",
            daemon=True,
        )
        self.web_thread.start()

        web_deadline = time.monotonic() + 30
        while time.monotonic() < web_deadline:
            if self.web_server.started:
                return
            if not self.web_thread.is_alive():
                self.process.terminate()
                raise RuntimeError("FastAPI wrapper failed during startup")
            time.sleep(0.1)
        self.web_server.should_exit = True
        self.web_thread.join(timeout=5)
        self.process.terminate()
        raise RuntimeError("FastAPI wrapper startup timed out")

    @modal.exit()
    def stop(self):
        import subprocess

        web_server = getattr(self, "web_server", None)
        web_thread = getattr(self, "web_thread", None)
        if web_server is not None:
            web_server.should_exit = True
        if web_thread is not None and web_thread.is_alive():
            web_thread.join(timeout=10)

        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
