"""Standalone HTTP delivery for per-request hidden-state accumulations.

This module deliberately owns its route and wire format instead of extending the
existing HS/QK delivery implementation.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import struct
import uuid
from typing import Any

import torch
from safetensors.torch import load, save

ROUTE_PATH = "/v1/mia/hidden-state-accumulation"
WIRE_FORMAT = "mia-hidden-state-accum-v1"
_DELIVERY_ID = re.compile(r"^[0-9a-f]{32}$")
logger = logging.getLogger(__name__)


def encode_result(result: dict) -> bytes:
    """Encode per-layer sums, means and counts as a compact safetensors payload."""
    tensors = {}
    layers = result.get("layers") or {}
    for layer, values in layers.items():
        layer = int(layer)
        summed = torch.as_tensor(
            values["sum"], device="cpu", dtype=torch.float32
        ).detach().contiguous()
        count = int(values["count"])
        mean = summed / count if count else torch.zeros_like(summed)
        tensors[f"{layer}/sum"] = summed
        tensors[f"{layer}/mean"] = mean.contiguous()
        tensors[f"{layer}/count"] = torch.tensor(count, dtype=torch.int64)
    metadata = {
        "format": WIRE_FORMAT,
        "layers": json.dumps(sorted(int(layer) for layer in layers)),
        "layer_names": json.dumps({
            str(int(layer)): str(name)
            for layer, name in (result.get("layer_names") or {}).items()
        }),
    }
    return save(tensors, metadata=metadata)


def decode_result(data: bytes) -> dict:
    """Decode :func:`encode_result` into a per-layer result mapping."""
    if len(data) < 8:
        raise ValueError("accumulation payload is truncated")
    header_size = struct.unpack("<Q", data[:8])[0]
    if header_size > len(data) - 8:
        raise ValueError("accumulation payload metadata is truncated")
    header = json.loads(data[8:8 + header_size].decode("utf-8"))
    metadata = header.get("__metadata__") or {}
    if metadata.get("format") != WIRE_FORMAT:
        raise ValueError("unsupported accumulation payload format")
    tensors = load(data)
    layers = {}
    for raw_layer in json.loads(metadata["layers"]):
        layer = int(raw_layer)
        layers[layer] = {
            "sum": tensors[f"{layer}/sum"],
            "mean": tensors[f"{layer}/mean"],
            "count": int(tensors[f"{layer}/count"].item()),
        }
    names = {int(layer): name for layer, name in
             json.loads(metadata["layer_names"]).items()}
    return {"layers": layers, "layer_names": names}


def new_delivery_id() -> str:
    """Create the opaque ID to pass as ``learning_delivery_id`` with a request."""
    return uuid.uuid4().hex


def _authorized(request) -> bool:
    """Apply the same bearer key configured for the vLLM server to this route."""
    expected = os.environ.get("VLLM_API_KEY")
    supplied = request.headers.get("authorization", "")
    return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")


# Check the worker for completed sums and counts until they are ready, then
# return them to the extraction service.
async def _wait_for_result(engine, delivery_id: str, timeout_s: float) -> dict | None:
    """Poll the worker RPC briefly to cover the scheduler-finish/HTTP race."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        results = await engine.collective_rpc(
            "get_accumulated_hidden_states", args=(delivery_id,)
        )
        found = [value for value in (results or ()) if isinstance(value, dict)]
        if len(found) > 1:
            raise RuntimeError("multiple worker ranks returned results; TP delivery is unsupported")
        if found:
            return found[0]
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(0.1, remaining))


def build_router():
    """Build the isolated accumulator read endpoint."""
    from fastapi import APIRouter, Query, Request
    from fastapi.responses import JSONResponse, Response

    router = APIRouter()

    async def read_accumulation(
        request: Any,
        delivery_id: str = Query(..., min_length=32, max_length=32),
        timeout_s: float = Query(30.0, gt=0.0, le=120.0),
    ):
        if not _authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not _DELIVERY_ID.fullmatch(delivery_id):
            return JSONResponse({"error": "invalid delivery_id"}, status_code=400)
        stage = "worker_rpc"
        try:
            result = await _wait_for_result(
                request.app.state.engine_client, delivery_id, timeout_s
            )
            if result is None:
                return JSONResponse(
                    {"error": "accumulation result is not ready"}, status_code=504
                )
            stage = "payload_encoding"
            return Response(encode_result(result), media_type="application/octet-stream")
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Accumulator result delivery failed at stage=%s delivery_id=%s",
                stage,
                delivery_id,
            )
            return JSONResponse(
                {
                    "error": "accumulation delivery failed",
                    "stage": stage,
                    "exception": type(exc).__name__,
                },
                status_code=500,
            )

    # With postponed annotations, the function's local Request import would
    # otherwise remain the unresolved string "Request" when FastAPI inspects it.
    read_accumulation.__annotations__["request"] = Request
    router.add_api_route(ROUTE_PATH, read_accumulation, methods=["GET"])
    return router


def register() -> None:
    """vLLM general-plugin entry point that attaches this route without editing core."""
    import vllm.entrypoints.launchers.app as app_module

    original_attach = app_module.attach_endpoint_plugins
    if getattr(original_attach, "_mia_accum_delivery", False):
        return

    def attach_endpoint_plugins(app, supported_tasks):
        original_attach(app, supported_tasks)
        app.include_router(build_router())

    attach_endpoint_plugins._mia_accum_delivery = True
    app_module.attach_endpoint_plugins = attach_endpoint_plugins


def fetch_result(base_url: str, api_key: str, delivery_id: str, timeout_s: float = 90.0) -> dict:
    """Fetch and decode one result from a vLLM server using its bearer API key."""
    import httpx

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    response = httpx.get(
        root + ROUTE_PATH,
        params={"delivery_id": delivery_id},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return decode_result(response.content)