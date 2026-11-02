import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "steer_routing_gpu_test", ROOT / "mia/core/hooks/steer_routing_gpu.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scatter_routes_distinct_vectors_to_each_layer():
    query_starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    slot_vectors = torch.tensor([[2, 0, 0], [0, 5, 0]], dtype=torch.int64)
    slot_modes = torch.zeros_like(slot_vectors)
    slot_coefficients = torch.tensor([[0.7, 0.0, 0.0], [0.0, -0.3, 0.0]])
    slot_layers = slot_vectors != 0
    coefficients = torch.zeros((3, 4))
    vector_ids = torch.zeros((3, 4), dtype=torch.int64)
    modes = torch.zeros_like(vector_ids)

    MODULE.scatter_routing(
        query_starts, slot_vectors, slot_modes, slot_coefficients, slot_layers,
        coefficients, vector_ids, modes, real_n=4, width=4, backend="torch")

    torch.testing.assert_close(vector_ids, torch.tensor([
        [2, 2, 0, 0], [0, 0, 5, 5], [0, 0, 0, 0],
    ]))
    torch.testing.assert_close(coefficients, torch.tensor([
        [0.7, 0.7, 0.0, 0.0], [0.0, 0.0, -0.3, -0.3], [0.0, 0.0, 0.0, 0.0],
    ]))


def test_scatter_preserves_legacy_shared_vector_slots():
    query_starts = torch.tensor([0, 2], dtype=torch.int32)
    slot_vectors = torch.tensor([4], dtype=torch.int64)
    slot_modes = torch.tensor([0], dtype=torch.int64)
    slot_coefficients = torch.tensor([0.25])
    slot_layers = torch.tensor([[True, False, True]])
    coefficients = torch.zeros((3, 2))
    vector_ids = torch.zeros((3, 2), dtype=torch.int64)
    modes = torch.zeros_like(vector_ids)

    MODULE.scatter_routing(
        query_starts, slot_vectors, slot_modes, slot_coefficients, slot_layers,
        coefficients, vector_ids, modes, real_n=2, width=2, backend="torch")

    torch.testing.assert_close(vector_ids, torch.tensor([[4, 4], [0, 0], [4, 4]]))
    torch.testing.assert_close(coefficients, torch.tensor([
        [0.25, 0.25], [0.0, 0.0], [0.25, 0.25],
    ]))