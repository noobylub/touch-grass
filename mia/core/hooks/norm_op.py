"""CUDA-graph-compatible per-token L2 norm of a decoder-layer output.

For Llama/Qwen-style tuple outputs, IBM captures hidden + residual. We do too.
Target: IBM/vLLM-Hook main, vLLM 0.29.0 V2 runner.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import Library

_REGISTERED = False
_LIB = None


@triton.jit
def _norm_kernel(hidden_states_ptr, residual_states_ptr, route_gate_ptr, scores_ptr,
                 num_tokens, hidden_size, hidden_token_stride, hidden_feature_stride,
                 residual_token_stride, residual_feature_stride, includes_residual: tl.constexpr,
                 BLOCK: tl.constexpr):
    token_index = tl.program_id(0)
    if token_index >= num_tokens:
        return
    should_score_token = tl.load(route_gate_ptr + token_index) != 0
    if should_score_token:
        squared_norm_accumulator = tl.full((BLOCK,), 0, tl.float32)
        for feature_offset in range(0, hidden_size, BLOCK):
            feature_index = feature_offset + tl.arange(0, BLOCK)
            hidden_value = tl.load(
                hidden_states_ptr + token_index * hidden_token_stride +
                feature_index * hidden_feature_stride,
                feature_index < hidden_size, other=0).to(tl.float32)
            if includes_residual:
                residual_value = tl.load(
                    residual_states_ptr + token_index * residual_token_stride +
                    feature_index * residual_feature_stride,
                    feature_index < hidden_size, other=0).to(tl.float32)
                hidden_value = hidden_value + residual_value
            squared_norm_accumulator += hidden_value * hidden_value
        token_norm = tl.sqrt(tl.sum(squared_norm_accumulator, axis=0))
        tl.store(scores_ptr + token_index, token_norm)
    else:
        tl.store(scores_ptr + token_index, 0.0)


def _impl(hidden_states: torch.Tensor, residual_states: torch.Tensor,
          route_gate: torch.Tensor, scores: torch.Tensor,
          includes_residual: int) -> None:
    if hidden_states.device.type != "cuda":
        raise RuntimeError("The norm worker requires CUDA")
    if hidden_states.ndim != 2:
        raise RuntimeError("Expected [tokens, hidden] layer output")
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens > route_gate.numel() or num_tokens > scores.numel():
        raise RuntimeError("Norm routing/capture capacity exceeded")
    _norm_kernel[(num_tokens,)](
        hidden_states, residual_states, route_gate, scores,
        num_tokens, hidden_size,
        hidden_states.stride(0), hidden_states.stride(1),
        residual_states.stride(0), residual_states.stride(1),
        bool(includes_residual),
        BLOCK=512, num_warps=4)


def _fake(hidden_states, residual_states, route_gate, scores,
          includes_residual: int) -> None:
    return None


def register_norm_op() -> None:
    global _REGISTERED, _LIB
    if _REGISTERED:
        return
    _LIB = Library("mia", "FRAGMENT")
    from mia.core.hooks.ops import _import_direct_register_custom_op
    register = _import_direct_register_custom_op()
    register(op_name="norm_l2", op_func=_impl, fake_impl=_fake,
             mutates_args=["scores"], target_lib=_LIB, dispatch_key="CUDA")
    _REGISTERED = True


_WRAPPED: dict[type, object] = {}


def wrap_decoder_layer(cls: type) -> None:
    """Wrap a decoder layer to call the graph-resident norm op."""
    if cls in _WRAPPED:
        return
    original = cls.forward
    _WRAPPED[cls] = original

    def wrapped(module, *args, **kwargs):
        layer_output = original(module, *args, **kwargs)
        hook_buffers = getattr(module, "_mia_norm_host", None)
        if hook_buffers is not None:
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
                residual_states = layer_output[1] if (len(layer_output) > 1 and
                    isinstance(layer_output[1], torch.Tensor) and
                    layer_output[1].shape == hidden_states.shape) else hidden_states
                includes_residual = int(residual_states is not hidden_states)
            else:
                hidden_states = residual_states = layer_output
                includes_residual = 0
            torch.ops.mia.norm_l2(
                hidden_states, residual_states, hook_buffers["gate"],
                hook_buffers["scores"], includes_residual)
        return layer_output
    cls.forward = wrapped