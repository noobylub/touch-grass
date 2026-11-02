import ast
import asyncio
import contextlib
import contextvars
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import uuid
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch, name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modes(monkeypatch):
    errors = SimpleNamespace(MiaConfigurationError=RuntimeError, MiaRefusal=RuntimeError)
    monkeypatch.setitem(sys.modules, "mia.errors", errors)
    mode = _load_module(monkeypatch, "mia.core.hooks.run_mode", "mia/core/hooks/run_mode.py")
    selector = _load_module(
        monkeypatch, "steer_capture_selector_test", "mia/core/delivery/delivery_selector.py")
    return mode, selector


@pytest.fixture
def plugin(monkeypatch, modes):
    for key in list(os.environ):
        if key.startswith("MIA_"):
            monkeypatch.delenv(key)
    mia_package = types.ModuleType("mia")
    mia_package.__path__ = []
    workers_package = types.ModuleType("mia.workers")
    workers_package.__path__ = []
    inline_module = types.ModuleType("mia.workers.steer_capture_inline")
    async def pin(*args):
        return None
    async def unpin(*args):
        return None
    inline_module.pin_steering_vector = pin
    inline_module.unpin_steering_vector = unpin
    monkeypatch.setitem(sys.modules, "mia", mia_package)
    monkeypatch.setitem(sys.modules, "mia.workers", workers_package)
    monkeypatch.setitem(sys.modules, "mia.workers.steer_capture_inline", inline_module)
    names = {
        "_kind_from_extension", "_worker_kind", "_enabled_subsystems", "mia_graph_layout",
        "_steering_only_for_delivery", "_aperture_per_request_mode",
        "_aperture_per_request_kind", "_takes_aperture_per_request", "_hybrid_serves",
        "_engine_worker_kind", "_offline_hybrid_request", "_offline_delivery_info",
        "_patched_generate", "_patched_create_engine_config",
    }
    tree = ast.parse((ROOT / "mia/core/_plugin.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith("_WORKER_")
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {
        "os": os, "STAMP_ENV": modes[1].STAMP_ENV,
        "_COMPILE_CACHE_STAMP_KEY": "mia_compile_cache",
        "parse_mia_worker_env": modes[0].parse_mia_worker_env,
        "DEFAULT_MIA_WORKER": modes[0].DEFAULT_MIA_WORKER,
        "MiaWorkerConflictError": RuntimeError,
        "_profile_mode": lambda: False,
        "_graph_mode": lambda: True,
        "_engine_graph": lambda engine: engine.graph,
        "_resolve_sink": lambda extra: extra.get("sink", "rpc"),
        "merge_info": lambda parts: parts[0],
        "resolve_hs_shard_mode": lambda tp: "round_robin",
        "HS_MODE_ROUND_ROBIN": "round_robin", "HS_MODE_RANK0": "rank0",
        "HS_MODE_ALL_RANKS": "all_ranks", "HS_SHARD_ENV": "MIA_HS_SHARD",
        "HS_ALL_RANKS_ENV": "MIA_HS_ALL_RANKS", "HS_LAYER_SHARD_RULE": "modulo",
    }
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "mia/core/_plugin.py"), "exec"),
         namespace)
    return namespace


def _engine(kind, graph=True):
    return SimpleNamespace(graph=graph, vllm_config=SimpleNamespace(
        additional_config={"mia_compile_cache": {"worker": kind}}))


def test_registration_uses_combined_worker_class():
    tree = ast.parse((ROOT / "mia/__init__.py").read_text())
    registrations = []
    registry = SimpleNamespace(
        register_worker=lambda name, worker: registrations.append((name, worker)),
        register_analyzer=lambda *args: None)
    combined = type("SteerCaptureWorker", (), {})
    namespace = {"PluginRegistry": registry, "SteerCaptureWorker": combined}
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "register_plugins")
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            namespace.setdefault(node.id, object())
    exec(compile(ast.Module(body=[function], type_ignores=[]), "mia/__init__.py", "exec"),
         namespace)
    namespace["register_plugins"]()
    assert ("steer_capture", combined) in registrations
    assert any(isinstance(node, ast.ImportFrom)
               and node.module == "mia.workers.steer_capture_worker"
               and node.names[0].name == "SteerCaptureWorker" for node in tree.body)
    exports = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__all__"
                           for target in node.targets))
    assert "SteerCaptureWorker" in ast.literal_eval(exports)


@pytest.mark.parametrize("env,graph,expected", [
    ({}, True, "hybrid"), ({"MIA_APERTURE_DELIVERY": "hybrid"}, True, "hybrid"),
    ({"MIA_APERTURE_PER_REQUEST": "1"}, True, "hybrid"),
    ({"MIA_APERTURE_DELIVERY": "drain"}, True, "drain"),
    ({}, False, "off"), ({"MIA_PROFILE_MODE": "1"}, True, "off"),
    ({"MIA_SINK": "drop"}, True, "off"),
])
def test_combined_selector_is_hs_producer(modes, env, graph, expected):
    mode, selector = modes
    assert mode.parse_mia_worker_env("steer_capture") == "steer_capture"
    env = {**env, "MIA_WORKER": "steer_capture"}
    assert selector.resolve(env, graph=graph).mode == expected


def test_kind_sizing_and_cache_include_both_subsystems(plugin, monkeypatch):
    extension = "mia.workers.steer_capture_worker.SteerCaptureWorker"
    assert plugin["_kind_from_extension"](extension) == "steer_capture"
    assert plugin["_kind_from_extension"](type("SteerCaptureWorker", (), {})) == "steer_capture"
    assert plugin["_worker_kind"](extension) == "steer_capture"
    assert plugin["_enabled_subsystems"]("steer_capture") == {"hs", "steer"}
    assert plugin["_enabled_subsystems"](["steer_capture", "qk"]) == {"hs", "steer", "qk"}
    args = SimpleNamespace(tensor_parallel_size=2)
    layout = plugin["mia_graph_layout"](args, "steer_capture")
    assert layout["kind"] == "steer_capture"
    assert layout["hs_layout"] == "round_robin"
    assert layout["steer_v_max"] == 16
    monkeypatch.setenv("MIA_STEER_VMAX", "32")
    assert plugin["mia_graph_layout"](args, "steer_capture") != layout
    monkeypatch.setenv("MIA_HS_CAPTURE", "alternate")
    assert plugin["mia_graph_layout"](args, "steer_capture")["hs_capture_mode"] == "alternate"
    monkeypatch.setenv("MIA_WORKER", "steer")
    with pytest.raises(RuntimeError, match="contradicts"):
        plugin["_worker_kind"](extension)


@pytest.mark.parametrize("kind,hs,qk,steer,allowed", [
    ("steer_capture", True, False, True, True),
    ("steer_capture", True, False, False, True),
    ("steer_capture", False, False, True, False),
    ("steer_capture", True, True, True, False),
    ("steer", True, False, True, False),
    ("hidden_states", True, False, True, False),
    ("hidden_states", True, False, False, True),
])
def test_delivery_gates(plugin, monkeypatch, kind, hs, qk, steer, allowed):
    engine = _engine(kind)
    extra = {}
    if hs:
        extra["output_hidden_states"] = [1]
    if qk:
        extra["output_qk"] = {1: [0]}
    if steer:
        extra["steer"] = {"vector_path": "vector.pt"}
    monkeypatch.setenv("MIA_APERTURE_PER_REQUEST", "1")
    assert (plugin["_aperture_per_request_kind"](extra, hs, qk, steer, engine) == "hs") == allowed
    monkeypatch.setenv("MIA_APERTURE_DELIVERY_RESOLVED", "hybrid:default")
    assert plugin["_hybrid_serves"](extra, hs, qk, steer, engine) == allowed
    assert plugin["_offline_hybrid_request"](extra, engine) == allowed
    extra["sink"] = "drop"
    assert plugin["_aperture_per_request_kind"](extra, hs, qk, steer, engine) is None
    assert not plugin["_hybrid_serves"](extra, hs, qk, steer, engine)


def test_offline_info_fetches_combined_hs_metadata(plugin):
    calls = []
    llm = SimpleNamespace(llm_engine=_engine("steer_capture"), collective_rpc=lambda name: (
        calls.append(name) or [{"roots": ["capture-root"]}]))
    assert plugin["_offline_delivery_info"](llm) == {"roots": ["capture-root"]}
    plugin["_offline_delivery_info"](llm)
    assert calls == ["mia_delivery_info"]


def test_startup_selects_combined_worker_and_sizes_both_subsystems(plugin, monkeypatch):
    events = []
    monkeypatch.setenv("MIA_WORKER", "steer_capture")
    monkeypatch.setitem(sys.modules, "mia.core.hooks.install", SimpleNamespace(
        set_graph_mode=lambda graph: events.append(("graph", graph))))
    args = SimpleNamespace(worker_extension_cls=None, compilation_config={})
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode="FULL"))
    mode = SimpleNamespace(graph=True, engine_eager=False, prefix_caching_off=False)
    plugin.update({
        "refuse_pipeline_parallel": lambda *args: None,
        "_note_vllm_version": lambda: None, "_STARTED_STAMPED": False,
        "_revert_mia_writes": lambda: None, "_ENGINE_HINTS": SimpleNamespace(get=lambda: None),
        "resolve_capture_mode": lambda *args, **kwargs: mode,
        "stamp_compile_cache_key": lambda args, kind: events.append(("cache", kind)),
        "_cc_mode_name": lambda cc: "FULL", "_default_cudagraph_mode": lambda cc: ("FULL", None),
        "_with_default_cudagraph": lambda cc, default: cc,
        "_capture_mode_line": lambda *args: "test mode", "_FULL_MODE_WARNING": "test full",
        "_apply_engine_delivery": lambda kind, graph: events.append(("delivery", kind, graph)),
        "_original_create_engine_config": lambda *args, **kwargs: config,
        "_cudagraph_mode_name": lambda name: name,
        "validate_graph_mode": lambda mode: None, "validate_v2_runner_selected": lambda cfg: None,
        "_maybe_autocap_max_batched_tokens": lambda cfg, kinds: events.append(
            ("sizing", plugin["_enabled_subsystems"](kinds))),
    })
    assert plugin["_patched_create_engine_config"](args) is config
    assert args.worker_extension_cls == "mia.workers.steer_capture_worker.SteerCaptureWorker"
    assert ("cache", "steer_capture") in events
    assert ("delivery", "steer_capture", True) in events
    assert ("sizing", {"hs", "steer"}) in events


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("kind,hs,delivery", [
    ("steer_capture", True, "drain"), ("steer_capture", True, "hybrid"),
    ("steer_capture", True, "eager"), ("steer_capture", True, "profile"),
    ("steer_capture", False, "drain"), ("steer", False, "drain"),
    ("steer", True, "drain"),
])
def test_finished_request_delivers_and_cleans_up(plugin, monkeypatch, kind, hs, delivery,
                                               interrupted):
    events = []
    engine = _engine(kind, graph=delivery != "eager")
    engine._mia_installed = True
    output = SimpleNamespace(finished=not interrupted, outputs=[SimpleNamespace(token_ids=[3])],
                             prompt_token_ids=[1, 2])
    extra = {"steer": {"vector_path": "vector.pt"}, "save_to_disk": False}
    if hs:
        extra["output_hidden_states"] = [1]
    params = SimpleNamespace(extra_args=extra, n=1)

    async def rpc(name, **kwargs):
        events.append(name)
        return [b"states"] if name == "get_captured_states" else None

    async def original(self, prompt, sampling_params, request_id, **kwargs):
        assert sampling_params.extra_args["steer"] == extra["steer"]
        yield output
        if interrupted:
            raise RuntimeError("stream interrupted")

    async def refuse(*args):
        return None

    async def await_hs(*args):
        events.append("await_hs")
        return {"hs_cache": {}}

    engine.collective_rpc = rpc
    plugin.update({
        "EngineCoreRequest": type("EngineCoreRequest", (), {}), "_json": json, "_ast": ast,
        "_refuse_unsupported_tp_request": lambda *args: None,
        "_refuse_on_graph_serve": refuse, "_sample_keys": lambda *args: ["req"],
        "_warn_profile_aperture_conflict": lambda: None,
        "_decide_aperture_route": lambda *args: None,
        "_aperture_finalize_action": lambda *args: "rpc_raw",
        "_await_aperture_per_request": await_hs,
        "_original_generate": original, "_hold_steer": lambda *args: None,
        "_release_steer": lambda *args, **kwargs: None,
        "_ADDED_IDS": contextvars.ContextVar("integration_added", default=None),
        "uuid": uuid, "time": time, "_SAMPLE_CACHED_ATTR": "_cached",
        "_end_completion_turn": lambda *args: events.append("end_turn"),
        "PROF": SimpleNamespace(incr=lambda name: events.append(name),
                                event=lambda *args: None,
                                timed=lambda *args: contextlib.nullcontext()),
        "_hs_ask_is_empty": lambda *args: False, "_hs_num_layers": lambda *args: 2,
        "_hybrid_marker": lambda *args: {"delivery": "hybrid"},
        "route_enabled": lambda: True, "_decompress": lambda payload: {"hs_cache": {}},
        "_raise_delivery_errors": lambda *args: None,
        "_reconstruct_compact_qk": lambda *args: None,
        "merge_probe_parts": lambda parts: parts[0], "_trim_probes": lambda *args: None,
        "_profile_mode": lambda: delivery == "profile",
        "_emit_capture_evidence": lambda *args: events.append("capture_evidence"),
    })
    if delivery == "hybrid":
        monkeypatch.setenv("MIA_APERTURE_DELIVERY_RESOLVED", "hybrid:default")
    else:
        monkeypatch.setenv("MIA_APERTURE_PER_REQUEST", "1")

    async def consume():
        return [item async for item in plugin["_patched_generate"](engine, "prompt", params, "req")]

    if interrupted:
        with pytest.raises(RuntimeError, match="stream interrupted"):
            asyncio.run(consume())
    else:
        assert asyncio.run(consume()) == [output]
        assert "steer.fire" in events
    combined_capture = kind == "steer_capture" and hs
    assert ("clear_captured_states" in events) == (combined_capture and delivery != "hybrid")
    assert "end_turn" in events
    if interrupted:
        assert "await_hs" not in events
        assert not hasattr(output, "probes")
        return
    if combined_capture:
        if delivery == "hybrid":
            assert output.mia == {"delivery": "hybrid"}
            assert "get_captured_states" not in events
        elif delivery == "profile":
            assert "capture_evidence" in events
        else:
            assert output.probes == {"hs_cache": {}}
            assert ("await_hs" in events) == (delivery == "drain")
        assert ("clear_aperture_request" in events) == (delivery in ("drain", "profile"))
    else:
        assert "await_hs" not in events
        assert not hasattr(output, "probes")