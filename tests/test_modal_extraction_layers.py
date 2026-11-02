import pytest

from deployment import modal_extraction
from client.extracting_vectors import layers_from_env
from deployment.modal_extraction import (
    resolve_requested_layers,
    resolve_vector_name,
    unique_vector_name,
)


def test_request_layers_override_configured_default():
    assert resolve_requested_layers({"layers": [10, 21]}, "10,21") == [10, 21]


def test_omitted_request_layers_use_configured_default():
    assert resolve_requested_layers({}, "10,21") == "10,21"


def test_optional_vector_name_is_trimmed():
    assert resolve_vector_name({}) is None
    assert resolve_vector_name({"vector_name": "  anger-study  "}) == "anger-study"


@pytest.mark.parametrize("value", [None, "", "  ", 42, "x" * 121])
def test_invalid_vector_names_are_rejected(value):
    with pytest.raises(ValueError):
        resolve_vector_name({"vector_name": value})


def test_duplicate_vector_name_gets_random_suffix():
    name = unique_vector_name("anger-study", 10, {"anger-study-layer-10"})
    assert name.startswith("anger-study-layer-10-")
    assert name not in {"anger-study-layer-10"}


def test_repeated_named_vector_is_saved_as_a_new_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_extraction, "VECTOR_DIR", str(tmp_path))
    monkeypatch.setattr(modal_extraction, "MODEL", "test-model")

    first_id, first_created, first_name = modal_extraction.store_extracted_vector(
        [0.1, 0.2], 10, "anger-study"
    )
    second_id, second_created, second_name = modal_extraction.store_extracted_vector(
        [0.1, 0.2], 10, "anger-study"
    )

    assert first_created and second_created
    assert first_id != second_id
    assert first_name == "anger-study-layer-10"
    assert second_name.startswith("anger-study-layer-10-")
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_client_layer_override_parses_environment_value():
    assert layers_from_env("10, 21") == [10, 21]
    assert layers_from_env(None) is None


@pytest.mark.parametrize("value", ["", "x,21", "-1,21"])
def test_client_rejects_invalid_environment_layer_value(value):
    with pytest.raises(ValueError):
        layers_from_env(value)


@pytest.mark.parametrize("layers", [[], "10,21", None, [10, True], [10, 21.0]])
def test_invalid_request_layers_are_rejected(layers):
    with pytest.raises(ValueError, match="non-empty list of integer layer indexes"):
        resolve_requested_layers({"layers": layers}, "10,21")