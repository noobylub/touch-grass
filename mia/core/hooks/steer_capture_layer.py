"""Ordered decoder wrapping for steering followed by raw hidden-state capture."""
from __future__ import annotations

import torch


_WRAPPED_LAYER_CLASSES: dict[type, object] = {}


def wrap_steer_capture_layer(cls: type) -> None:
    """Install both graph-resident operations once, with capture after steering."""
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    original_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = original_forward

    def forward(module, *args, **kwargs):
        output = original_forward(module, *args, **kwargs)
        steer_host = getattr(module, "_mia_steer_host", None)
        if steer_host is not None and steer_host.do_steer:
            if (isinstance(output, tuple) and len(output) >= 2
                    and isinstance(output[1], torch.Tensor)):
                steer_host.steer(output[1])
            elif isinstance(output, torch.Tensor):
                steer_host.steer(output)

        capture_host = getattr(module, "_mia_hs_host", None)
        if capture_host is not None and capture_host.do_capture:
            if (isinstance(output, tuple) and len(output) >= 2
                    and isinstance(output[0], torch.Tensor)
                    and isinstance(output[1], torch.Tensor)):
                torch.ops.mia.capture_hs(
                    output[0], output[1], capture_host.hs_buf,
                    capture_host.capture_index, 1,
                )
            else:
                hidden = output[0] if isinstance(output, tuple) else output
                if isinstance(hidden, torch.Tensor):
                    torch.ops.mia.capture_hs(
                        hidden, hidden, capture_host.hs_buf,
                        capture_host.capture_index, 0,
                    )
        return output

    cls.forward = forward