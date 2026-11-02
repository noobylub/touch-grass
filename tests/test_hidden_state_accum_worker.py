from collections import OrderedDict
import json

import torch

from mia.core.delivery.hidden_state_accum_delivery import decode_result, encode_result
from mia.workers.hidden_state_accum_worker import HiddenStateAccumWorker


def _worker_with_cpu_accumulators():
    worker = HiddenStateAccumWorker()
    worker.num_layers = 1
    worker.hidden_size = 2
    worker.request_capacity = 2
    worker.accumulated_sums = torch.zeros((1, 2, 2), dtype=torch.float32)
    worker.accumulated_token_counts = torch.zeros((1, 2), dtype=torch.int64)
    worker._request_to_slot = {}
    worker._free_slots = [1, 0]
    worker._request_layers = {}
    worker._next_token_position = {}
    worker._request_group_ids = {}
    worker._groups = {}
    worker._completed_group_ids = OrderedDict()
    worker._completed_hidden_states = OrderedDict()
    worker._request_delivery_ids = {}
    worker._completed_request_delivery_ids = {}
    worker._completed_delivery_results = OrderedDict()
    worker._layer_names = {0: "model.layers.0"}
    return worker


def _finish_request(worker, request_id, token_sum, token_count):
    slot = worker._get_or_allocate_slot(request_id)
    worker._request_layers[request_id] = {0}
    worker.accumulated_sums[0, slot] = torch.tensor(token_sum)
    worker.accumulated_token_counts[0, slot] = token_count
    worker._finish_request(request_id)


def test_group_result_waits_for_all_requests_and_averages_request_means():
    worker = _worker_with_cpu_accumulators()
    group_config = {
        "learning_group_id": "person-1",
        "learning_group_size": 2,
        "learning_delivery_id": "a" * 32,
    }

    worker._register_group_request("input-1", group_config)
    _finish_request(worker, "input-1", token_sum=[6.0, 12.0], token_count=3)
    assert worker.get_accumulated_hidden_states("a" * 32) is None

    worker._register_group_request("input-2", group_config)
    _finish_request(worker, "input-2", token_sum=[30.0, 6.0], token_count=2)

    result = worker.get_accumulated_hidden_states("a" * 32)
    assert result is not None
    assert result["layers"][0]["count"] == 2
    assert isinstance(result["layers"][0]["sum"], list)
    decoded = decode_result(encode_result(json.loads(json.dumps(result))))
    torch.testing.assert_close(
        decoded["layers"][0]["sum"], torch.tensor([17.0, 7.0])
    )

    # The input vectors are [2, 4] and [15, 3]; each input has equal weight.
    torch.testing.assert_close(
        decoded["layers"][0]["mean"],
        torch.tensor([8.5, 3.5]),
    )


def test_group_size_must_be_positive():
    worker = _worker_with_cpu_accumulators()
    config = {
        "learning_group_id": "person-1",
        "learning_group_size": 0,
        "learning_delivery_id": "a" * 32,
    }

    try:
        worker._register_group_request("input-1", config)
    except ValueError as exc:
        assert "learning_group_size" in str(exc)
    else:
        raise AssertionError("expected invalid group size to be rejected")