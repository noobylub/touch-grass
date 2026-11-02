import ast
import asyncio
import importlib.util
from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "steer_capture_inline_test", ROOT / "mia/workers/steer_capture_inline.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def engine():
    instance = SimpleNamespace(model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(hidden_size=2, num_hidden_layers=4),
        dtype=torch.float16))
    yield instance
    storage = getattr(instance, "_mia_inline_vector_storage", None)
    if storage is not None:
        storage.cleanup()


def config(**overrides):
    return {"vector": [1.0, 2.0], "method": "add_vector", "optimal_layer": 1, **overrides}


def test_roundtrip_and_reuse(engine):
    calls = []
    resolve = lambda: MODULE.resolve_inline_steering(
        engine, config(), lambda engine, path: calls.append(path))
    first = resolve()
    assert resolve() == first
    assert len(list(Path(engine._mia_inline_vector_storage.name).iterdir())) == 1
    assert len(calls) == 2
    assert "vector" not in first
    torch.testing.assert_close(torch.load(first["vector_path"], weights_only=True)["dir"],
                               torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("overrides", [
    {"vector": [1]}, {"vector": [[1], [2]]}, {"vector": [True, 1]},
    {"vector": [float("nan"), 1]}, {"vector": [1e10, 1]},
    {"optimal_layer": 4}, {"optimal_layer": []}, {"vector_path": "vector.pt"},
    {"method": "invalid"}, {"coefficient": float("inf")},
    {"phase": "invalid"}, {"positions": "invalid"},
    {"method": "adjust_rs"}, {"method": "adjust_rs", "avg_proj": 1},
])
def test_invalid_request_does_not_allocate_storage(engine, overrides):
    with pytest.raises(ValueError):
        MODULE.resolve_inline_steering(engine, config(**overrides), lambda *args: None)
    assert not hasattr(engine, "_mia_inline_vector_storage")


def test_capacity_rejection_does_not_write(engine):
    def reject(*args):
        raise ValueError("capacity")

    with pytest.raises(ValueError, match="capacity"):
        MODULE.resolve_inline_steering(engine, config(), reject)
    assert not list(Path(engine._mia_inline_vector_storage.name).iterdir())


def test_adjust_rs(engine):
    result = MODULE.resolve_inline_steering(
        engine, config(vector=[1, 0], method="adjust_rs", avg_proj=0.5), lambda *args: None)
    assert torch.load(result["vector_path"], weights_only=True)["avg_proj"].item() == 0.5


def test_fifo_eviction_skips_active_and_reuses_oldest_idle(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("MIA_STEER_VMAX", "2")
    storage = tempfile.TemporaryDirectory(dir=tmp_path)
    engine._mia_inline_vector_storage = storage
    engine._mia_steer_fifo_entries = OrderedDict((
        ("old.pt", 1), ("next.pt", 0)))
    evicted = []

    async def rpc(method, args):
        assert method == "evict_steering_vector"
        evicted.append(args[0])
        return [True]

    engine.collective_rpc = rpc
    engine._mia_steer_interned = {"old.pt": True, "next.pt": True}
    asyncio.run(MODULE.pin_steering_vector(engine, "new.pt"))
    assert list(engine._mia_steer_fifo_entries.items()) == [
        ("old.pt", 1), ("new.pt", 1)]
    assert evicted == ["next.pt"]
    assert "next.pt" not in engine._mia_steer_interned


def test_fifo_refcounts_duplicate_requests_and_rejects_when_all_active(
        engine, monkeypatch):
    monkeypatch.setenv("MIA_STEER_VMAX", "1")
    async def rpc(*args, **kwargs):
        pytest.fail("an active vector must not be evicted")

    engine.collective_rpc = rpc
    asyncio.run(MODULE.pin_steering_vector(engine, "shared.pt"))
    asyncio.run(MODULE.pin_steering_vector(engine, "shared.pt"))
    assert engine._mia_steer_fifo_entries["shared.pt"] == 2
    asyncio.run(MODULE.unpin_steering_vector(engine, "shared.pt"))
    assert engine._mia_steer_fifo_entries["shared.pt"] == 1
    with pytest.raises(RuntimeError, match="all steering vector slots are in use"):
        asyncio.run(MODULE.pin_steering_vector(engine, "other.pt"))
    asyncio.run(MODULE.unpin_steering_vector(engine, "shared.pt"))
    assert engine._mia_steer_fifo_entries["shared.pt"] == 0


def test_registry_eviction_recycles_slot_and_invalidates_routing():
    tree = ast.parse((ROOT / "mia/core/hooks/install_steer.py").read_text())
    registry_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                          and node.name == "SteerRegistry")
    method = next(node for node in registry_class.body if isinstance(node, ast.FunctionDef)
                  and node.name == "evict_path")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "SteerRegistry.evict_path", "exec"),
         namespace)
    registry = SimpleNamespace(
        vec_paths={"old.pt": 0}, vec_table=torch.tensor([[1.0, 2.0]]),
        avg_proj_table=torch.tensor([0.5]), vec_has_avgproj={0}, _free_vec_ids=[],
        _st_intern={(0, 1.0, 0, 0): 1}, _st_val={1: (0, 1.0, 0, 0)},
        _skip_warned={"old.pt"},
    )
    registry.force_full_routing = lambda: setattr(registry, "routing_invalidated", True)
    worker = SimpleNamespace(_vector_cache={"old.pt": {"dir": torch.ones(2)}})

    assert namespace["evict_path"](registry, "old.pt", worker)

    assert registry.vec_paths == {}
    assert registry._free_vec_ids == [0]
    assert torch.equal(registry.vec_table[0], torch.zeros(2))
    assert torch.equal(registry.avg_proj_table, torch.zeros(1))
    assert not registry.vec_has_avgproj
    assert worker._vector_cache == {}
    assert not registry._st_intern and not registry._st_val
    assert registry.routing_invalidated


@pytest.mark.parametrize("kind", ["steer_capture", "steer", "hidden_states"])
def test_served_admission_is_scoped(monkeypatch, engine, kind):
    tree = ast.parse((ROOT / "mia/core/_plugin.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_refuse_on_graph_serve")
    monkeypatch.setitem(sys.modules, "mia.workers.steer_capture_inline", MODULE)
    namespace = {
        "_engine_worker_kind": lambda engine: kind, "MiaConfigurationError": ValueError,
        "_check_steer_vmax": lambda *args: None, "_engine_graph": lambda engine: False,
        "_refuse_on_graph": lambda engine, extra, **kwargs: extra["steer"],
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "served_admission", "exec"),
         namespace)
    request = {"steer": config()}
    if kind != "steer_capture":
        with pytest.raises(ValueError, match="only by steer_capture"):
            asyncio.run(namespace[function.name](engine, request))
    else:
        result = asyncio.run(namespace[function.name](engine, request))
        assert "vector_path" in result and "vector" not in result