# Worker Development Guide

This guide describes how to add a worker to this repository and make it selectable in offline `MiaLLM` and deployed `vllm serve`. It is separate from the project `README.md`.

## How a worker is selected

MIA selects one worker extension per engine; registering a worker does not make all workers run together.

For `vllm serve`, `MIA_WORKER` selects the class through `mia/core/_plugin.py`. The class is attached by vLLM as `worker_extension_cls`. For offline use, `MiaLLM(worker_name=...)` gets a dotted class path from `PluginRegistry`.

In CUDA-graph mode, MIA's model-load patch calls the selected worker's `graph_install()` after the model is loaded. Eager-mode features use `install_hooks()` if supported by that worker. Keep installation idempotent because engine setup can call lifecycle code more than once.

## Add a worker

1. **Implement the worker class** under `mia/workers/`, following a nearby worker's mixin pattern. Keep GPU work in graph-safe fixed-buffer/custom-op paths when using CUDA graphs. Expose RPC-callable methods as worker methods; vLLM's `collective_rpc("method_name", args=(...))` dispatches to them.
2. **Register it for offline use** in `mia/__init__.py`: import the class and add `PluginRegistry.register_worker("name", ClassName)` inside `register_plugins()`.
3. **Register the served worker kind** in `mia/core/hooks/run_mode.py` by adding the exact `MIA_WORKER` value to `MIA_WORKER_VALUES`.
4. **Map the kind to its class** in `mia/core/_plugin.py` by adding the dotted class path to `_WORKER_EXT_BY_KIND`. The worker kind is also used by compile-cache layout, delivery selection, and resource sizing; inspect nearby kind-dependent branches and add explicit behavior where needed. Do not let a new kind silently inherit hidden-state behavior.
5. **Add worker-specific startup behavior only where needed.** For example, refuse unsupported parallelism explicitly rather than returning incomplete data.
6. **Add tests** for config parsing, worker selection, operation/result behavior, and failure paths. GPU graph replay needs a CUDA test environment.

Example registration:

```python
# mia/__init__.py
from mia.workers.example_worker import ExampleWorker

# Inside register_plugins():
PluginRegistry.register_worker("example", ExampleWorker)
```

```python
# mia/core/hooks/run_mode.py
MIA_WORKER_VALUES = ("hidden_states", "hidden_state_accum", "qk", "steer", "example")
```

```python
# mia/core/_plugin.py
_WORKER_EXT_BY_KIND["example"] = "mia.workers.example_worker.ExampleWorker"
```

## Configure a deployment

Set the worker kind in the deployment environment, for example:

```python
.env({
    "MIA_WORKER": "hidden_state_accum",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
})
```

Pass per-request options via `vllm_xargs`. For the accumulator worker:

```python
extra_body={"vllm_xargs": {
    "learning_enable": True,
    "learning_delivery_id": delivery_id,
    "learning_group_id": "person-42",
    "learning_group_size": 200,
    "learning_layers": [0, 10, 21],
    "learning_phase": "both",
    "learning_positions": "all_tokens",
}}
```

For grouped extraction, give every input in the group the same `learning_group_id`,
`learning_group_size`, and `learning_delivery_id`. The group ID identifies membership; the delivery
ID is the opaque identifier used to fetch the completed result. The worker's existing CUDA hook
averages the selected token vectors per input, then adds each input mean to the group's GPU
accumulator. After all 200 requests finish, the endpoint returns one equal-weight average vector
per selected layer (the result's per-layer `count` is the number of contributing inputs). Without
`learning_group_id`, the worker keeps its existing per-request result behavior. Do not include
legacy `output_hidden_states` unless you intend to run the separate hidden-state capture path.

After changing worker selection or vLLM plugin entry points, rebuild/reinstall the editable package in the deployment image and redeploy it. Changing only request xargs does not change which worker the engine instantiates.

## Expose results over HTTP

Worker capture and HTTP delivery are separate responsibilities. A worker can retain results and expose an RPC method, but clients cannot call worker RPC directly through the OpenAI API.

For a worker-specific result endpoint:

1. Put its route, payload codec, and client fetch helper in a focused module under `mia/core/delivery/` (or another clearly named package location). Keep it separate from existing HS/QK formats unless the result shape truly matches them.
2. Have the endpoint validate/authenticate the request, accept only an opaque delivery ID, and call the worker through `request.app.state.engine_client.collective_rpc(...)`.
3. Return a versioned binary tensor format. Enforce a bounded wait and return a not-ready status rather than blocking indefinitely. Keep result lookup retry-safe until delivery is confirmed, and bound retained results to avoid unbounded worker memory.
4. Register the route from the existing `mia.core._plugin:register` lifecycle alongside the shared delivery route. Do not add a second vLLM general-plugin entry point for each worker unless there is a concrete reason for an independently loadable plugin.
5. Provide a client decoder/fetch helper and tests for round-trip encoding, invalid IDs, missing results, retry behavior, and supported parallelism.

The current accumulator route follows this pattern at `/v1/mia/hidden-state-accumulation`; it returns per-layer `sum`, `mean`, and `count`. Its current delivery implementation supports a single worker rank only.

## Before deployment

- Confirm `MIA_WORKER` is accepted and maps to the intended class.
- Confirm the worker's graph install succeeds on the target vLLM version and GPU.
- Test one request with a known delivery ID and verify layer selection, token count, result shape, and endpoint authentication.
- Test missing/not-yet-completed IDs and ensure the endpoint returns a bounded timeout.
- Rebuild the Modal image after changing package or plugin registration.
