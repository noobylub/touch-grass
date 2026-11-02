"""Generate with activation steering and return raw post-steering hidden states."""
from __future__ import annotations

from mia.errors import MiaConfigurationError
from mia.core.hooks.registry import get_registry
from mia.workers.hs_capture_worker import HSCaptureWorker


class SteerCaptureWorker(HSCaptureWorker):
    """Combined graph worker; inherits raw HS retrieval and delivery methods."""

    def install_hooks(self):
        raise MiaConfigurationError(
            "steer_capture requires CUDA-graph mode; remove enforce_eager=True "
            "and set MIA_ALLOW_CUDAGRAPH=1."
        )

    def graph_install(self):
        from mia.core.hooks.install_steer_capture import install_steer_capture

        install_steer_capture(self)

    def evict_steering_vector(self, path: str) -> bool:
        """RPC: clear an idle steering vector from this worker's GPU table."""
        registry = get_registry(self, "steer")
        return bool(registry and registry.evict_path(path, self))