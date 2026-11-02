"""CPU checks for the ordering of the combined graph operations."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "mia/core/hooks/steer_capture_layer.py"
SPEC = importlib.util.spec_from_file_location("steer_capture_layer_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("steer_enabled,capture_enabled", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_capture_follows_steering(monkeypatch, tuple_output, steer_enabled, capture_enabled):
    events = []
    captured = []
    hidden = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])
    output = (hidden, residual) if tuple_output else hidden

    class Decoder:
        def forward(self):
            events.append("decoder")
            return output

    def steer(values):
        events.append("steer")
        values.add_(10)

    def capture(hidden_states, residual_states, buffer, indices, includes_residual):
        events.append("capture")
        values = hidden_states + residual_states if includes_residual else hidden_states
        captured.append(values.clone())

    monkeypatch.setattr(torch.ops.mia, "capture_hs", capture, raising=False)
    decoder = Decoder()
    decoder._mia_steer_host = SimpleNamespace(do_steer=steer_enabled, steer=steer)
    decoder._mia_hs_host = SimpleNamespace(
        do_capture=capture_enabled, hs_buf=None, capture_index=None,
    )
    MODULE.wrap_steer_capture_layer(Decoder)
    MODULE.wrap_steer_capture_layer(Decoder)
    assert decoder.forward() is output
    assert events == ["decoder"] + (["steer"] if steer_enabled else []) + (
        ["capture"] if capture_enabled else []
    )
    if capture_enabled:
        expected = torch.tensor([[4.0, 6.0]]) if tuple_output else torch.tensor([[1.0, 2.0]])
        if steer_enabled:
            expected += 10
        torch.testing.assert_close(captured[0], expected)


def test_layer_without_hosts_preserves_output():
    output = torch.ones(1, 2)

    class Decoder:
        def forward(self):
            return output

    MODULE.wrap_steer_capture_layer(Decoder)
    assert Decoder().forward() is output