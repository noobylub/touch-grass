import numpy as np
import torch

from mia.core.delivery.hidden_state_accum_delivery import decode_result, encode_result


def test_accumulation_delivery_round_trip():
    result = {
        "layers": {
            3: {"sum": torch.tensor([3.0, 6.0]), "count": 3},
            7: {"sum": torch.tensor([8.0, 4.0]), "count": 2},
        },
        "layer_names": {3: "model.layers.3", 7: "model.layers.7"},
    }

    decoded = decode_result(encode_result(result))

    assert decoded["layer_names"] == result["layer_names"]
    assert decoded["layers"][3]["count"] == 3
    torch.testing.assert_close(decoded["layers"][3]["sum"], torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(decoded["layers"][3]["mean"], torch.tensor([1.0, 2.0]))
    assert decoded["layers"][7]["count"] == 2
    torch.testing.assert_close(decoded["layers"][7]["mean"], torch.tensor([4.0, 2.0]))


def test_accumulation_delivery_accepts_rpc_numeric_representations():
    for summed in (torch.tensor([6.0, 12.0]), np.array([6.0, 12.0]), [6.0, 12.0]):
        result = {
            "layers": {3: {"sum": summed, "count": 3}},
            "layer_names": {3: "model.layers.3"},
        }
        decoded = decode_result(encode_result(result))
        assert decoded["layers"][3]["count"] == 3
        torch.testing.assert_close(
            decoded["layers"][3]["mean"], torch.tensor([2.0, 4.0])
        )
