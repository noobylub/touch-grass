"""CPU checks for combined installation and public request defaults."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text())
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == class_name)
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef)
                  and node.name == method_name)
    namespace = {"Optional": Optional, "Dict": Dict, "json": json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
    return namespace[method_name]


def test_combined_installer_coordinates_subsystems_once(monkeypatch):
    events = []
    wrapper = object()

    def install_steer(worker, *, layer_wrapper):
        assert layer_wrapper is wrapper
        events.append("steer")

    def install_hs(worker, *, layer_wrapper):
        assert layer_wrapper is wrapper
        events.append("hs")

    def install_drain(runner, worker):
        assert runner is worker.model_runner
        events.append("drain")

    monkeypatch.setitem(sys.modules, "mia.core.hooks.steer_capture_layer",
                        SimpleNamespace(wrap_steer_capture_layer=wrapper))
    monkeypatch.setitem(sys.modules, "mia.core.hooks.install_steer",
                        SimpleNamespace(install_steer_hosts=install_steer))
    monkeypatch.setitem(sys.modules, "mia.core.hooks.install_hs", SimpleNamespace(
        install_hs_hosts=install_hs, install_execute_model_wrapper_hs=install_drain))
    spec = importlib.util.spec_from_file_location(
        "combined_install_test", ROOT / "mia/core/hooks/install_steer_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = SimpleNamespace(model_runner=object())
    module.install_steer_capture(worker)
    module.install_steer_capture(worker)
    assert events == ["steer", "hs", "drain"]


@pytest.mark.parametrize("kind,has_hs,has_steer", [
    ("steer_capture", True, True), ("capture_hs", True, False),
    ("steer", False, True), ("capture_qk", False, False),
])
def test_offline_defaults(kind, has_hs, has_steer):
    build = load_method("mia/llm.py", "MiaLLM", "_build_extra_args")
    worker = SimpleNamespace(
        worker_name=kind, _output_layers=[1, 3], _hs_mode="all_tokens",
        _steering_config={"method": "add_vector", "coefficient": 1},
        layer_to_heads={}, _hookq_mode="all_tokens", _qk_capture="qk",
        _hook_dir="unused",
    )
    extra = build(worker, False, "run", {"steer": {"coefficient": 2}})
    assert ("output_hidden_states" in extra) is has_hs
    assert ("steer" in extra) is has_steer
    if has_steer:
        assert extra["steer"] == {"method": "add_vector", "coefficient": 2}


@pytest.mark.parametrize("with_capture", [True, False])
def test_client_mixed_config_retains_capture(tmp_path, with_capture):
    config = {"steering": {"method": "add_vector"}}
    if with_capture:
        config["hidden_states"] = {"layers": [1], "mode": "all_tokens"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    client = SimpleNamespace()
    load_method("mia/client.py", "MiaClient", "_load_config")(client, str(path))
    extra = load_method("mia/client.py", "MiaClient", "_build_extra_body")(client)
    assert ("output_hidden_states" in extra["vllm_xargs"]) is with_capture