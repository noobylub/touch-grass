import ast
import math
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deployment/modal_inference.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_build_multi_vector_steer")
NAMESPACE = {"math": math, "MAX_VECTORS": 16}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
build_steer = NAMESPACE[FUNCTION.name]
VECTOR_ID_ASSIGNMENT = next(
    node for node in TREE.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "VECTOR_ID_RE" for target in node.targets)
)
VECTOR_ID_NAMESPACE = {"re": re}
exec(compile(ast.Module(body=[VECTOR_ID_ASSIGNMENT], type_ignores=[]), str(SOURCE), "exec"), VECTOR_ID_NAMESPACE)
vector_id_pattern = VECTOR_ID_NAMESPACE["VECTOR_ID_RE"]


def entry(vector_id, layer, coefficient, stored_layer=None):
    return {
        "vector_id": vector_id,
        "vector_path": f"/vectors/{vector_id}.pt",
        "stored_layer": stored_layer,
        "requested_layer": layer,
        "coefficient": coefficient,
    }


def test_vector_ids_are_bare_sha256_hashes():
    assert vector_id_pattern.fullmatch("a" * 64)
    assert not vector_id_pattern.fullmatch(f"sha256:{'a' * 64}")


def test_builds_separate_vectors_at_their_resolved_layers():
    steer, metadata = build_steer([
        entry("sha256:a", None, 0.7, stored_layer=3),
        entry("sha256:b", 8, -0.3, stored_layer=6),
    ], num_layers=12)

    assert steer == {
        "method": "add_vector",
        "vectors": [
            {"vector_path": "/vectors/sha256:a.pt", "layer": 3, "coefficient": 0.7},
            {"vector_path": "/vectors/sha256:b.pt", "layer": 8, "coefficient": -0.3},
        ],
        "phase": "both",
        "positions": "all_tokens",
    }
    assert metadata == [
        {"vector_id": "sha256:a", "layer": 3, "vector_layer_id": 3},
        {"vector_id": "sha256:b", "layer": 8, "vector_layer_id": 6},
    ]


@pytest.mark.parametrize("entries", [
    [entry("sha256:a", 3, 0.7), entry("sha256:b", 3, -0.3)],
    [entry("sha256:a", 12, 0.7)],
    [entry("sha256:a", 3, float("nan"))],
])
def test_rejects_duplicate_or_invalid_layer_and_coefficient(entries):
    with pytest.raises(ValueError):
        build_steer(entries, num_layers=12)


def test_rejects_more_than_resident_vector_capacity():
    entries = [entry(f"sha256:{index}", index, 1.0) for index in range(17)]

    with pytest.raises(ValueError, match="1 to 16"):
        build_steer(entries, num_layers=20)