# Multi-Vector Steering

## Scope

Multi-vector steering applies multiple saved steering directions during one **served vLLM request** using the CUDA-graph steering path. Each vector is applied at its own transformer layer. A request may target a layer only once; it does not combine multiple vectors at the same layer.

This feature does not change eager steering or offline inference. The existing single-vector request remains supported.

## Request Format

Send either the existing `vector_id` field or the new `vectors` field to `/steer`, but not both:

```json
{
  "vectors": [
    {"vector_id": "sha256:<first-vector-id>", "coefficient": 0.7},
    {"vector_id": "sha256:<second-vector-id>", "coefficient": -0.3}
  ],
  "prompt": "Your prompt here",
  "max_tokens": 128
}
```

Each stored vector's `layer_id` is used by default. A caller may override the layer for an individual vector:

```json
{"vector_id": "sha256:<vector-id>", "layer": 12, "coefficient": 0.7}
```

The endpoint resolves the IDs to saved vector files and sends vLLM a graph-routing config shaped like:

```json
{
  "method": "add_vector",
  "vectors": [
    {"vector_path": "/vectors/steering-vectors/<hash>.pt", "layer": 3, "coefficient": 0.7},
    {"vector_path": "/vectors/steering-vectors/<hash>.pt", "layer": 8, "coefficient": -0.3}
  ],
  "phase": "both",
  "positions": "all_tokens"
}
```

The per-vector `layer` is a single layer index. Duplicate target layers, invalid layer indices, non-finite coefficients, and lists exceeding `MAX_VECTORS` are rejected. The vector count is bounded by the endpoint's `MAX_VECTORS` and must fit the worker's resident-vector capacity (`MIA_STEER_VMAX`).

## Graph Routing

The worker loads each vector into its resident vector table and keeps a separate vector ID, coefficient, and method for each request/layer pair. The existing steering operation then applies the one selected vector for that layer. The GPU scatter path carries per-layer values; legacy single-vector configs continue to broadcast their one vector to their configured layer set.

Per-entry `adjust_rs` is not part of this multi-vector request format. The endpoint and graph router use additive steering (`add_vector`) for vector-list requests.

## Architecture Decisions

- **One vector per layer:** The routing output already has one vector ID, coefficient, and method for every `(layer, token-column)` cell. Multi-vector requests therefore expand into several layer entries; they do not need a second vector-slot dimension in the steering op or a new steering kernel.
- **Keep vectors separate:** Each stored direction stays in the worker's resident vector table and is selected by its own ID at its target layer. The endpoint does not sum directions into one vector, because that would lose the per-layer placement and independent coefficients.
- **Use saved layer metadata:** Extracted artifacts include `layer_id`, so that is the default target and keeps extraction and inference aligned. An explicit per-item `layer` override is available for callers that intentionally want a different layer.
- **Adapt only the GPU scatter input:** The GPU fast path previously kept one vector ID and coefficient per request and broadcast them to selected layers. It now stores those values per request and layer, then scatters into the same existing `[layer, token-column]` graph tables. The downstream per-layer steering operation remains unchanged. The legacy shared-vector slot form remains supported.
- **Keep the feature served and graph-only:** The new public request is handled at the served endpoint and consumed by the CUDA-graph routing path. Eager steering and offline inference retain their existing behavior and interfaces.
- **Reject ambiguous requests:** A request cannot target one layer more than once. Rejecting duplicates avoids silently choosing an order for multiple additive vectors at the same layer, which is outside the one-vector-per-layer contract.
- **Bound resident state:** The request list is limited by `MAX_VECTORS` and must fit `MIA_STEER_VMAX`, since each distinct saved direction occupies a resident worker vector-table entry.

## Implemented

- `deployment/modal_inference.py` accepts either the legacy single-vector request or a multi-vector list, resolves saved layer metadata, validates the list, and reports the applied vector IDs and layers.
- `mia/core/hooks/install_steer.py` normalizes legacy and multi-vector configs and routes list entries to their individual graph layers.
- `mia/core/hooks/steer_routing_gpu.py` supports per-request/per-layer IDs and coefficients while retaining the legacy shared-slot layout.
- `tests/test_modal_inference_multi_vector.py` covers endpoint config construction and input validation.
- `tests/test_steer_multi_graph_routing.py` covers per-layer routing and legacy scatter compatibility.

## Remaining Verification

- Run a served integration test on a CUDA host using the deployed vLLM/vLLM-Hook image and CUDA graphs.
- Verify a request with two vectors at distinct saved layers is steered at both layers, and verify a single-vector request still behaves as before.
- Exercise the GPU Triton scatter kernel on that CUDA host; local CPU tests cover the Torch scatter fallback only.
- Confirm Modal volume reload and resident-vector capacity behavior with multiple vector IDs, including a request at the configured maximum.

## Out of Scope

- Multiple vectors targeting the same layer.
- Eager worker multi-vector support.
- Offline inference changes.
- Changes to vector extraction or saved-vector format; extraction already stores `layer_id` with each vector.
