# In this file, I will learn how to implement my own worker, and write my learning in a documentation 

import torch

from mia.core.runner import require_v2_runner, install_request_arg_stash, step_view
from mia.workers._common import iter_matched_modules, match_layer


# We want to determine the span of tokens that are relevant for learning for each request.
def _targeted_span(step_view, request_index, learning_config):
    """Return a scheduled-token slice (or None) for this request."""
    query_start_locations = step_view.query_start_loc_np
    token_start = int(query_start_locations[request_index])
    token_end = int(query_start_locations[request_index + 1])
    if token_end <= token_start or str(learning_config.get("learning_enable", "0")).lower() in ("0", "false"):
        return None
    learning_phase = str(learning_config.get("learning_phase", "both"))
    if learning_phase not in ("prefill", "decode", "both"):
        raise ValueError("Invalid learning_phase")
    is_prefill_step = bool(step_view.is_prefilling_np[request_index])
    if (learning_phase == "prefill" and not is_prefill_step) or (learning_phase == "decode" and is_prefill_step):
        return None
    learning_positions = str(learning_config.get("learning_positions", "last_token"))
    if learning_positions == "all_tokens":
        return token_start, token_end
    if learning_positions != "last_token":
        raise ValueError("Invalid learning_positions")
    if is_prefill_step and (int(step_view.num_computed_tokens_np[request_index]) +
                            token_end - token_start <
                            int(step_view.prompt_len_np[request_index])):
        return None
    return token_end - 1, token_end

# For this tutorial worker, I will perform the norm of the targeted span's or request's activation
class TutorialWorker:
    def install_hooks(self):
        raise NotImplementedError("This does not support installed eager hooks")

    # hooking up the wrapper, this would be done in another file in deployment setting 
    def graph_install(self):
        if getattr(self, "_graph_installed", False):
            return

        from mia.core.hooks.install import _resolve_max_num_batched_tokens
        from mia.core.hooks.registry import PinnedMirror
        from mia.core.hooks.norm_op import register_norm_op, wrap_decoder_layer


        # We retrieve the model and hook it up 
        # Make sure it has the necessary attributes for hooking
        runner = self.model_runner
        require_v2_runner(runner)
        model = runner.model 

        # Register the CUDA op before the wrapped layer tries to call it.
        register_norm_op()

        # Size routing/output storage for the largest token batch vLLM may schedule.
        self.cap = _resolve_max_num_batched_tokens(self)
        device = next(model.parameters()).device

        # The graph reads one gate value per token and writes one norm score per token.
        self.route = torch.zeros(self.cap, device=device, dtype=torch.bool)
        self.scores = torch.zeros(self.cap, device=device, dtype=torch.float32)

        # CPU-pinned staging memory lets each step upload routing values asynchronously.
        self.mirror = PinnedMirror([("route", (self.cap,), torch.bool)], pin=True)

        # Find exactly the first decoder block; this tutorial instruments only that layer.
        matched = [(n, m, l) for n,m,l in iter_matched_modules(model, match_layer)
                    if l == 0]
        if len(matched) != 1:
            raise RuntimeError("Expected exactly one layer 0")
        name, module, _ = matched[0]

        # Give the wrapper the GPU buffers, then add the norm op call to this layer's forward.
        # We also wrap the layer here
        module._mia_norm_host = {"gate": self.route, "scores": self.scores}
        wrap_decoder_layer(type(module))

        # Preserve request-specific extra_args and keep the original input-preparation method
        # so the next block can add routing without replacing vLLM's behavior.
        stash = install_request_arg_stash(runner)
        prepare = runner.prepare_inputs

        # Wrap vLLM's input preparation: keep its batch unchanged, but on ordinary
        # steps snapshot request metadata and refresh the GPU token gate before the
        # decoder-layer wrapper runs the norm operation.
        def prepare_inputs(*args, **kwargs):
            batch = prepare(*args, **kwargs)
            if torch.cuda.is_current_stream_capturing():
                # Leave routing alone while CUDA is capturing the graph.
                return batch

            step = step_view(runner, batch, stash)
            self._route(step)
            return batch

        runner.prepare_inputs = prepare_inputs
        self._graph_installed = True

    def _route(self, step):
        """Build the per-token gate read by the norm hook for this scheduler step."""
        self.mirror.wait_current()
        cpu = self.mirror.cur("route")
        cpu.zero_()  # Prevent the next user inheriting stale settings.
        # Mark selected request spans, then upload this mask as the norm op's gate.
        for i in range(step.num_reqs):
            cfg = step.extra_args_for(i) or {}
            span = _targeted_span(step, i, cfg)
            if span is not None:
                cpu[span[0]:span[1]] = 1
        self.route.copy_(cpu, non_blocking=True)
        self.mirror.record_advance()
