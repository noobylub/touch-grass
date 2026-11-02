"""CUDA-graph worker that accumulates selected hidden states per request."""
from __future__ import annotations

from collections import OrderedDict

import torch

from mia.core.runner import require_v2_runner, install_request_arg_stash, step_view
from mia.workers._common import iter_matched_modules, match_layer


def _parse_learning_layers(value, num_layers: int) -> list[int]:
    """Normalize an integer, sequence, or comma-separated layer selection."""
    if isinstance(value, int):
        requested_layers = [value]
    elif isinstance(value, (list, tuple)):
        requested_layers = [int(layer) for layer in value]
    else:
        requested_layers = [
            int(layer.strip())
            for layer in str(value).split(",")
            if layer.strip()
        ]

    selected_layers = sorted(set(requested_layers))
    if not selected_layers or any(
        layer < 0 or layer >= num_layers for layer in selected_layers
    ):
        raise ValueError(
            f"learning_layers must select indexes from 0 to {num_layers - 1}; "
            f"received {selected_layers!r}"
        )
    return selected_layers


def _targeted_span(step, request_index: int, learning_config: dict):
    """Return selected scheduled-token rows, defaulting to every text token."""
    query_start_locations = step.query_start_loc_np
    token_start = int(query_start_locations[request_index])
    token_end = int(query_start_locations[request_index + 1])
    if token_end <= token_start or str(
        learning_config.get("learning_enable", "0")
    ).lower() in ("0", "false"):
        return None

    learning_phase = str(learning_config.get("learning_phase", "both"))
    if learning_phase not in ("prefill", "decode", "both"):
        raise ValueError("learning_phase must be prefill, decode or both")
    is_prefill_step = bool(step.is_prefilling_np[request_index])
    if (learning_phase == "prefill" and not is_prefill_step) or (
        learning_phase == "decode" and is_prefill_step
    ):
        return None

    # Unlike the norm example, accumulation defaults to all scheduled tokens.
    learning_positions = str(learning_config.get("learning_positions", "all_tokens"))
    if learning_positions == "all_tokens":
        return token_start, token_end
    if learning_positions != "last_token":
        raise ValueError("learning_positions must be all_tokens or last_token")

    # Do not treat a partial prefill chunk as the prompt's last token.
    if is_prefill_step and (
        int(step.num_computed_tokens_np[request_index]) + token_end - token_start
        < int(step.prompt_len_np[request_index])
    ):
        return None
    return token_end - 1, token_end


class HiddenStateAccumWorker:
    """Accumulate hidden-vector sums and counts for each enabled request."""

    _MAX_DELIVERY_RESULTS = 128

    def install_hooks(self):
        raise NotImplementedError("This worker requires the CUDA Graph path")

    def graph_install(self):
        """Allocate stable buffers and install the graph-resident decoder hook."""
        if getattr(self, "_hidden_accum_graph_installed", False):
            return

        from mia.core.hooks.install import (
            _resolve_max_num_batched_tokens,
            _resolve_max_num_seqs,
        )
        from mia.core.hooks.registry import PinnedMirror
        from mia.core.hooks.hidden_state_accum_op import (
            register_hidden_accumulation_op,
            wrap_decoder_layer_for_accumulation,
        )

        runner = self.model_runner
        require_v2_runner(runner)
        model = runner.model
        register_hidden_accumulation_op()

        # Token routing capacity and concurrent request capacity are different:
        # one request may contribute many token rows in the same scheduler step.
        self.token_capacity = _resolve_max_num_batched_tokens(self)
        self.request_capacity = _resolve_max_num_seqs(self)
        if self.request_capacity is None:
            raise RuntimeError("Could not resolve max_num_seqs for request accumulators")

        model_config = getattr(model.config, "text_config", model.config)
        self.hidden_size = int(model_config.hidden_size)
        self.num_layers = int(model_config.num_hidden_layers)
        model_device = next(model.parameters()).device

        # These buffers stay at fixed addresses so the captured graph can update
        # them on every replay. Sums use FP32 regardless of model activation dtype.
        self.accumulated_sums = torch.zeros(
            (self.num_layers, self.request_capacity, self.hidden_size),
            device=model_device,
            dtype=torch.float32,
        )
        self.accumulated_token_counts = torch.zeros(
            (self.num_layers, self.request_capacity),
            device=model_device,
            dtype=torch.int64,
        )
        self.token_slots = torch.full(
            (self.num_layers, self.token_capacity),
            -1,
            device=model_device,
            dtype=torch.int64,
        )
        self.token_slot_mirror = PinnedMirror(
            [("token_slots", (self.num_layers, self.token_capacity), torch.int64)],
            pin=True,
        )

        # CPU bookkeeping maps transient vLLM request IDs to reusable GPU rows.
        self._request_to_slot: dict[str, int] = {}
        self._free_slots = list(reversed(range(self.request_capacity)))
        self._request_layers: dict[str, set[int]] = {}
        self._next_token_position: dict[tuple[str, int], int] = {}
        self._request_group_ids: dict[str, str] = {}
        self._groups: dict[str, dict] = {}
        self._completed_group_ids: OrderedDict[str, None] = OrderedDict()
        self._completed_hidden_states: OrderedDict[str, dict] = OrderedDict()
        self._request_delivery_ids: dict[str, str] = {}
        self._completed_request_delivery_ids: dict[str, str] = {}
        self._completed_delivery_results: OrderedDict[str, dict] = OrderedDict()

        # Each layer gets views of its own request sums, counts, and token map.
        matched_layers = list(iter_matched_modules(model, match_layer))
        layers_by_index = {layer_index: (name, module)
                           for name, module, layer_index in matched_layers}
        expected_layer_indexes = set(range(self.num_layers))
        if set(layers_by_index) != expected_layer_indexes:
            raise RuntimeError(
                "Matched decoder layer indexes do not match model config: "
                f"matched={sorted(layers_by_index)}, expected="
                f"{sorted(expected_layer_indexes)}"
            )
        self._layer_names = {}
        for layer_index in range(self.num_layers):
            layer_name, layer_module = layers_by_index[layer_index]
            self._layer_names[layer_index] = layer_name
            # This attribute is the handoff to hidden_state_accum_op: the
            # wrapped forward reads these layer-specific buffers and calls the op.
            layer_module._mia_hidden_accum_host = {
                "token_slots": self.token_slots[layer_index],
                "sums": self.accumulated_sums[layer_index],
                "token_counts": self.accumulated_token_counts[layer_index],
            }
            wrap_decoder_layer_for_accumulation(type(layer_module))

        request_arg_stash = install_request_arg_stash(runner)
        original_prepare_inputs = runner.prepare_inputs

        def prepare_inputs(*args, **kwargs):
            batch = original_prepare_inputs(*args, **kwargs)
            if not torch.cuda.is_current_stream_capturing():
                scheduler_step = step_view(runner, batch, request_arg_stash)
                self._route_tokens(scheduler_step)
            return batch

        runner.prepare_inputs = prepare_inputs

        # The runner tells us when IDs are finished. Snapshot their result before
        # releasing the GPU row, then let the original finish wrapper clear args.
        original_finish_requests = runner.finish_requests

        def finish_requests(scheduler_output):
            finish_result = original_finish_requests(scheduler_output)
            for request_id in scheduler_output.finished_req_ids:
                self._finish_request(str(request_id))
            return finish_result

        runner.finish_requests = finish_requests
        self._hidden_accum_graph_installed = True
        print(
            f"[hidden-accum] installed on {self.num_layers} layers: "
            f"requests={self.request_capacity}, tokens={self.token_capacity}, "
            f"hidden={self.hidden_size}",
            flush=True,
        )

    def _get_or_allocate_slot(self, request_id: str) -> int:
        """Return a stable GPU row for this request, zeroing it on first use."""
        existing_slot = self._request_to_slot.get(request_id)
        if existing_slot is not None:
            return existing_slot
        if not self._free_slots:
            raise RuntimeError(
                "No free hidden-state accumulator slots; max_num_seqs was exceeded"
            )

        new_slot = self._free_slots.pop()
        self.accumulated_sums[:, new_slot].zero_()
        self.accumulated_token_counts[:, new_slot].zero_()
        self._request_to_slot[request_id] = new_slot
        return new_slot

    # Before each model pass, map selected tokens to the GPU row for their request.
    # The decoder hook uses this map to add each token to the right request totals.
    def _route_tokens(self, step):
        """Route selected token rows to request slots before the decoder forward.

        Each layer's row in this map is consumed by that layer's wrapped forward
        and passed to the CUDA accumulation op.
        """
        self.token_slot_mirror.wait_current()
        host_token_slots = self.token_slot_mirror.cur("token_slots")
        host_token_slots.fill_(-1)

        for request_index, request_id in enumerate(step.req_ids):
            learning_config = step.extra_args_for(request_index) or {}
            group_id = self._register_group_request(request_id, learning_config)
            selected_span = _targeted_span(step, request_index, learning_config)
            if selected_span is None:
                continue
            delivery_id = learning_config.get("learning_delivery_id")
            if delivery_id and group_id is None:
                delivery_id = str(delivery_id)
                if any(
                    group["delivery_id"] == delivery_id
                    for group in self._groups.values()
                ):
                    raise ValueError(f"learning_delivery_id {delivery_id!r} is already in use")
                previous_delivery_id = self._request_delivery_ids.get(request_id)
                if previous_delivery_id not in (None, delivery_id):
                    raise ValueError("learning_delivery_id changed during an active request")
                self._request_delivery_ids[request_id] = delivery_id
            selected_layers = _parse_learning_layers(
                learning_config.get("learning_layers", "0"), self.num_layers
            )

            token_start = int(step.query_start_loc_np[request_index])
            computed_tokens = int(step.num_computed_tokens_np[request_index])
            absolute_start = computed_tokens + selected_span[0] - token_start
            request_slot = self._get_or_allocate_slot(request_id)
            self._request_layers.setdefault(request_id, set()).update(selected_layers)

            for layer_index in selected_layers:
                previous_end = self._next_token_position.get(
                    (request_id, layer_index), absolute_start
                )

                # A preempted request can be recomputed from an earlier position.
                # Skip already accumulated rows separately for each layer.
                layer_start = selected_span[0]
                if absolute_start < previous_end:
                    duplicate_rows = min(
                        selected_span[1] - selected_span[0],
                        previous_end - absolute_start,
                    )
                    layer_start += duplicate_rows
                if layer_start >= selected_span[1]:
                    continue

                # The hook sees these scheduled-token columns and writes their
                # hidden states into this request's persistent GPU accumulator row.
                host_token_slots[layer_index, layer_start:selected_span[1]] = request_slot
                absolute_end = computed_tokens + selected_span[1] - token_start
                self._next_token_position[(request_id, layer_index)] = max(
                    previous_end, absolute_end
                )

        # Upload the [layer, scheduled-token-row] routing table. The wrapped
        # decoder layers read their own row during each graph replay.
        self.token_slots.copy_(host_token_slots, non_blocking=True)
        self.token_slot_mirror.record_advance()

    def _register_group_request(self, request_id: str, learning_config: dict) -> str | None:
        """Bind a prompt request to its group, independent of submission chunk."""
        raw_group_id = learning_config.get("learning_group_id")
        if raw_group_id is None:
            return None

        group_id = str(raw_group_id)
        if not group_id:
            raise ValueError("learning_group_id must not be empty")
        try:
            raw_group_size = learning_config["learning_group_size"]
            if isinstance(raw_group_size, bool):
                raise ValueError
            group_size = int(raw_group_size)
        except (KeyError, TypeError, ValueError):
            raise ValueError("learning_group_size must be a positive integer") from None
        if group_size < 1:
            raise ValueError("learning_group_size must be a positive integer")

        raw_delivery_id = learning_config.get("learning_delivery_id")
        if raw_delivery_id is None or not str(raw_delivery_id):
            raise ValueError(
                "learning_delivery_id is required when learning_group_id is set"
            )
        delivery_id = str(raw_delivery_id)

        request_group = self._request_group_ids.get(request_id)
        if request_group is not None:
            group = self._groups[request_group]
            if (request_group != group_id or group["expected_size"] != group_size
                    or group["delivery_id"] != delivery_id):
                raise ValueError("learning group settings changed during an active request")
            return group_id

        if group_id in self._completed_group_ids:
            raise ValueError(f"learning_group_id {group_id!r} has already completed")

        group = self._groups.get(group_id)
        if group is None:
            if delivery_id in self._completed_delivery_results:
                raise ValueError(f"learning_delivery_id {delivery_id!r} is already in use")
            if any(
                active_group["delivery_id"] == delivery_id
                for active_group in self._groups.values()
            ):
                raise ValueError(f"learning_delivery_id {delivery_id!r} is already in use")
            group = {
                "expected_size": group_size,
                "delivery_id": delivery_id,
                "members": set(),
                "finished": 0,
                "sums": torch.zeros(
                    (self.num_layers, self.hidden_size),
                    device=self.accumulated_sums.device,
                    dtype=torch.float32,
                ),
                "counts": torch.zeros(
                    (self.num_layers,),
                    device=self.accumulated_sums.device,
                    dtype=torch.int64,
                ),
            }
            self._groups[group_id] = group
        elif (group["expected_size"] != group_size
              or group["delivery_id"] != delivery_id):
            raise ValueError("requests in a learning group must share size and delivery ID")

        if len(group["members"]) >= group["expected_size"]:
            raise ValueError(f"learning group {group_id!r} has more members than declared")
        group["members"].add(request_id)
        self._request_group_ids[request_id] = group_id
        return group_id

    def _finish_group(self, group_id: str, request_slot: int | None) -> None:
        """Add one request mean on GPU and publish only after the group is complete."""
        group = self._groups[group_id]
        if request_slot is not None:
            request_counts = self.accumulated_token_counts[:, request_slot]
            has_values = request_counts > 0
            request_means = self.accumulated_sums[:, request_slot] / (
                request_counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
            )
            group["sums"].add_(request_means * has_values.unsqueeze(-1))
            group["counts"].add_(has_values.to(torch.int64))

        # Requests may finish out of order or across chunks; publish only when
        # every request registered for this group has completed.
        group["finished"] += 1
        if group["finished"] < group["expected_size"]:
            return
        if group["finished"] > group["expected_size"]:
            raise RuntimeError(f"learning group {group_id!r} finished more than once")

        # The group sum contains one per-request token mean per member, so its
        # count makes the existing delivery codec compute an equal-input mean.
        sums = group["sums"].detach().cpu()
        counts = group["counts"].detach().cpu()
        layers = {
            layer_index: {"sum": sums[layer_index], "count": int(counts[layer_index])}
            for layer_index in range(self.num_layers)
            if int(counts[layer_index]) > 0
        }
        delivery_id = group["delivery_id"]
        self._completed_delivery_results[delivery_id] = {
            "layers": layers,
            "layer_names": dict(self._layer_names),
        }
        self._completed_delivery_results.move_to_end(delivery_id)
        self._completed_group_ids[group_id] = None
        self._completed_group_ids.move_to_end(group_id)
        del self._groups[group_id]

        while len(self._completed_delivery_results) > self._MAX_DELIVERY_RESULTS:
            self._completed_delivery_results.popitem(last=False)
        while len(self._completed_group_ids) > self._MAX_DELIVERY_RESULTS:
            self._completed_group_ids.popitem(last=False)

    def _finish_request(self, request_id: str) -> None:
        """Snapshot one completed request and return its GPU row to the free pool."""
        request_slot = self._request_to_slot.pop(request_id, None)
        group_id = self._request_group_ids.pop(request_id, None)
        selected_layers = self._request_layers.pop(request_id, set())
        for layer_index in selected_layers:
            self._next_token_position.pop((request_id, layer_index), None)
        if request_slot is None:
            if group_id is not None:
                self._finish_group(group_id, None)
            return

        # This synchronization is intentionally at request completion, not in
        # the captured forward. The returned CPU tensor is independent of reuse.
        completed_count_by_layer = self.accumulated_token_counts[:, request_slot].detach().cpu()
        completed_layers = {}
        for layer_index in sorted(selected_layers):
            layer_count = int(completed_count_by_layer[layer_index])
            layer_sum = self.accumulated_sums[layer_index, request_slot].detach().cpu()
            completed_layers[layer_index] = {
                "sum": layer_sum,
                "count": layer_count,
            }
        self._completed_hidden_states[request_id] = {
            "layers": completed_layers,
        }
        self._completed_hidden_states.move_to_end(request_id)
        delivery_id = self._request_delivery_ids.pop(request_id, None)
        if delivery_id and group_id is None:
            self._completed_delivery_results[delivery_id] = {
                "layers": completed_layers,
                "layer_names": dict(self._layer_names),
            }
            self._completed_delivery_results.move_to_end(delivery_id)
            self._completed_request_delivery_ids[request_id] = delivery_id
            while len(self._completed_delivery_results) > self._MAX_DELIVERY_RESULTS:
                expired_delivery_id, _ = self._completed_delivery_results.popitem(last=False)
                expired_request = next(
                    (req for req, value in self._completed_request_delivery_ids.items()
                     if value == expired_delivery_id),
                    None,
                )
                if expired_request is not None:
                    self._completed_request_delivery_ids.pop(expired_request, None)
        if group_id is not None:
            self._finish_group(group_id, request_slot)
        while len(self._completed_hidden_states) > self._MAX_DELIVERY_RESULTS:
            expired_request_id, _ = self._completed_hidden_states.popitem(last=False)
            expired_delivery_id = self._completed_request_delivery_ids.pop(
                expired_request_id, None
            )
            if expired_delivery_id is not None:
                self._completed_delivery_results.pop(expired_delivery_id, None)
        self.accumulated_sums[:, request_slot].zero_()
        self.accumulated_token_counts[:, request_slot].zero_()
        self._free_slots.append(request_slot)

    def pop_accumulated_hidden_states(self, request_id: str) -> dict | None:
        """Pop a completed request's ``sum``, ``count`` and mean vector.

        This method synchronizes only indirectly through the completion snapshot,
        never from the decoder forward. It returns ``None`` if the ID is unknown
        or has not finished yet.
        """
        completed_result = self._completed_hidden_states.pop(str(request_id), None)
        if completed_result is None:
            return None

        layer_results = {}
        for layer_index, values in completed_result["layers"].items():
            accumulated_sum = values["sum"]
            token_count = values["count"]
            mean_hidden_state = (
                accumulated_sum / token_count
                if token_count
                else torch.zeros_like(accumulated_sum)
            )
            layer_results[layer_index] = {
                "sum": accumulated_sum,
                "count": token_count,
                "mean": mean_hidden_state,
            }
        return {
            "layers": layer_results,
        }

    def get_accumulated_hidden_states(self, delivery_id: str) -> dict | None:
        """RPC read using transport-safe numeric values; non-consuming for retries."""
        result = self._completed_delivery_results.get(str(delivery_id))
        if result is None:
            return None
        return {
            "layers": {
                int(layer): {
                    "sum": values["sum"].tolist(),
                    "count": int(values["count"]),
                }
                for layer, values in result["layers"].items()
            },
            "layer_names": dict(result["layer_names"]),
        }