"""CUDA-graph custom op for per-request hidden-state accumulation."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import Library

_REGISTERED = False
_LIB = None


@triton.jit
def _accumulate_hidden_kernel(
    hidden_states_ptr,
    residual_states_ptr,
    token_slots_ptr,
    accumulated_sums_ptr,
    token_counts_ptr,
    num_tokens,
    hidden_size,
    hidden_token_stride,
    hidden_feature_stride,
    residual_token_stride,
    residual_feature_stride,
    includes_residual: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program handles one token and one feature block. Tokens assigned to
    # the same request update the same persistent vector through atomic adds.
    token_index = tl.program_id(0)
    feature_block = tl.program_id(1)
    if token_index < num_tokens:
        request_slot = tl.load(token_slots_ptr + token_index)
        if request_slot >= 0:
            feature_indices = feature_block * BLOCK + tl.arange(0, BLOCK)
            feature_mask = feature_indices < hidden_size

            hidden_values = tl.load(
                hidden_states_ptr + token_index * hidden_token_stride +
                feature_indices * hidden_feature_stride,
                feature_mask,
                other=0,
            ).to(tl.float32)
            if includes_residual:
                residual_values = tl.load(
                    residual_states_ptr + token_index * residual_token_stride +
                    feature_indices * residual_feature_stride,
                    feature_mask,
                    other=0,
                ).to(tl.float32)
                hidden_values += residual_values

            sum_offset = request_slot * hidden_size + feature_indices
            tl.atomic_add(
                accumulated_sums_ptr + sum_offset,
                hidden_values,
                mask=feature_mask,
                sem="relaxed",
            )

            # One feature block contributes one count for this token, not one
            # count per feature block.
            if feature_block == 0:
                tl.atomic_add(token_counts_ptr + request_slot, 1, sem="relaxed")


def _accumulate_impl(
    hidden_states: torch.Tensor,
    residual_states: torch.Tensor,
    token_slots: torch.Tensor,
    accumulated_sums: torch.Tensor,
    token_counts: torch.Tensor,
    includes_residual: int,
) -> None:
    """Add each routed token vector into its request's sum and token count."""
    if hidden_states.device.type != "cuda":
        raise RuntimeError("Hidden-state accumulation requires CUDA")
    if hidden_states.ndim != 2:
        raise RuntimeError("Expected [tokens, hidden] layer output")
    if token_slots.numel() < hidden_states.shape[0]:
        raise RuntimeError("Hidden-state token-slot capacity exceeded")
    if accumulated_sums.ndim != 2 or accumulated_sums.shape[1] != hidden_states.shape[1]:
        raise RuntimeError("Expected [request slots, hidden] accumulation buffer")
    if token_counts.ndim != 1 or token_counts.shape[0] != accumulated_sums.shape[0]:
        raise RuntimeError("Token-count buffer must match accumulation request slots")
    if includes_residual and residual_states.shape != hidden_states.shape:
        raise RuntimeError("Residual and hidden states must have matching shapes")

    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return

    block_size = 128
    _accumulate_hidden_kernel[(num_tokens, triton.cdiv(hidden_size, block_size))](
        hidden_states,
        residual_states,
        token_slots,
        accumulated_sums,
        token_counts,
        num_tokens,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        residual_states.stride(0),
        residual_states.stride(1),
        bool(includes_residual),
        BLOCK=block_size,
        num_warps=4,
    )


def _accumulate_fake(
    hidden_states,
    residual_states,
    token_slots,
    accumulated_sums,
    token_counts,
    includes_residual: int,
) -> None:
    return None


def register_hidden_accumulation_op() -> None:
    """Register ``torch.ops.mia.accumulate_hidden`` once per worker process."""
    global _REGISTERED, _LIB
    if _REGISTERED:
        return

    _LIB = Library("mia", "FRAGMENT")
    from mia.core.hooks.ops import _import_direct_register_custom_op

    register = _import_direct_register_custom_op()
    register(
        op_name="accumulate_hidden",
        op_func=_accumulate_impl,
        fake_impl=_accumulate_fake,
        mutates_args=["accumulated_sums", "token_counts"],
        target_lib=_LIB,
        dispatch_key="CUDA",
    )
    _REGISTERED = True


_WRAPPED_DECODER_CLASSES: dict[type, object] = {}


# After a decoder layer runs, its selected token states are added to each
# request's running sum and count by the GPU operation below.
def wrap_decoder_layer_for_accumulation(cls: type) -> None:
    """Call the accumulation op after a decoder layer produces its output."""
    if cls in _WRAPPED_DECODER_CLASSES:
        return

    original_forward = cls.forward
    _WRAPPED_DECODER_CLASSES[cls] = original_forward

    def wrapped_forward(module, *args, **kwargs):
        layer_output = original_forward(module, *args, **kwargs)
        accumulation_buffers = getattr(module, "_mia_hidden_accum_host", None)
        if accumulation_buffers is not None:
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
                residual_states = (
                    layer_output[1]
                    if len(layer_output) > 1
                    and isinstance(layer_output[1], torch.Tensor)
                    and layer_output[1].shape == hidden_states.shape
                    else hidden_states
                )
                includes_residual = int(residual_states is not hidden_states)
            else:
                hidden_states = layer_output
                residual_states = layer_output
                includes_residual = 0

            # The worker fills this map before the model pass. A -1 means skip
            # that token; any other value selects the request's GPU buffer row.
            # Add the selected states to those buffers and update their counts.
            torch.ops.mia.accumulate_hidden(
                hidden_states,
                residual_states,
                accumulation_buffers["token_slots"],
                accumulation_buffers["sums"],
                accumulation_buffers["token_counts"],
                includes_residual,
            )
        return layer_output

    cls.forward = wrapped_forward