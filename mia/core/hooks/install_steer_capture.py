"""Coordinate steering and raw HS capture without duplicate decoder wrappers."""
from __future__ import annotations

from mia.core.hooks.steer_capture_layer import wrap_steer_capture_layer


def install_steer_capture(worker) -> None:
    """Reuse subsystem routing and HS delivery, capturing after local steering."""
    if getattr(worker, "_mia_steer_capture_installed", False):
        return

    from mia.core.hooks.install_hs import (
        install_execute_model_wrapper_hs,
        install_hs_hosts,
    )
    from mia.core.hooks.install_steer import install_steer_hosts

    install_steer_hosts(worker, layer_wrapper=wrap_steer_capture_layer)
    install_hs_hosts(worker, layer_wrapper=wrap_steer_capture_layer)
    install_execute_model_wrapper_hs(worker.model_runner, worker)
    worker._mia_steer_capture_installed = True