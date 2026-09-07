from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from minisgl.core import Batch, Req
from minisgl.engine import Engine
from minisgl.engine.config import EngineConfig

from .catalog import Catalog
from .search import BeamState, expand, remap_decode_window


@dataclass
class DecodeInput:
    batch: Batch
    state: BeamState
    steps: torch.Tensor
    rows: torch.Tensor
    prompt_lens: torch.Tensor
    noise: torch.Tensor


class DecodeGraphs:
    """Capture model + restricted head + trie search + KV-index remap in one replay."""

    def __init__(self, runtime, groups: int):
        self.runtime = runtime
        self.graphs, self.buffers, self.outputs = {}, {}, {}
        engine, width = runtime.engine, runtime.width
        backend = engine.attn_backend
        sizes = [n * width for n in range(1, groups + 1)]
        backend.init_capture_graph(engine.page_table.shape[1], sizes)
        pool = None
        for n in range(groups, 0, -1):
            count = n * width
            batch = Batch([engine.dummy_req] * count, "decode")
            batch.padded_reqs = batch.reqs
            device = engine.device
            batch.input_ids = torch.zeros(count, dtype=torch.int32, device=device)
            batch.positions = torch.zeros_like(batch.input_ids)
            batch.out_loc = torch.full_like(batch.input_ids, engine.num_pages)
            backend.prepare_for_capture(batch)
            # Distinct table rows avoid racing writes during warmup/capture.
            rows = torch.arange(count, device=device).view(n, width)
            buffers = DecodeInput(
                batch,
                runtime.initial_state(n, width),
                torch.zeros(n, dtype=torch.long, device=device),
                rows,
                torch.ones(n, dtype=torch.long, device=device),
                torch.zeros(n, width, runtime.catalog.max_degree, device=device),
            )
            with engine.ctx.forward_batch(batch):
                runtime._decode_step(buffers)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=engine.stream):
                    result = runtime._decode_step(buffers)
            pool = graph.pool()
            self.graphs[n], self.buffers[n], self.outputs[n] = graph, buffers, result
        engine.page_table[: groups * width].fill_(engine.num_pages)

    def replay(self, inputs: DecodeInput):
        n = inputs.steps.numel()
        buffers = self.buffers[n]
        for field in ("input_ids", "positions", "out_loc"):
            getattr(buffers.batch, field).copy_(getattr(inputs.batch, field))
        for target, source in zip(buffers.state, inputs.state):
            target.copy_(source)
        for field in ("steps", "rows", "prompt_lens", "noise"):
            getattr(buffers, field).copy_(getattr(inputs, field))
        self.runtime.engine.attn_backend.prepare_for_replay(inputs.batch)
        self.graphs[n].replay()
        # These buffers are overwritten on replay. Scheduler copies each group's state.
        return self.outputs[n]


class RecommendationRuntime:
    def __init__(
        self,
        config: EngineConfig,
        catalog: Catalog,
        width: int,
        max_requests: int,
        graphs: bool = True,
        fused_search: bool = False,
    ):
        if min(width, max_requests) < 1:
            raise ValueError("Beam width and max requests must be positive")
        if config.tp_info.size != 1 or config.page_size != 1 or config.model_config.is_moe:
            raise ValueError("Recommendation runtime requires one GPU, dense model, page_size=1")
        self.catalog, self.width = catalog, width
        self.vocab_size = config.model_config.vocab_size
        self.fused_search = fused_search
        config = replace(
            config,
            output_token_ids=catalog.token_ids,
            cuda_graph_bs=[],
            max_running_req=width * max_requests,
            attention_backend="fi",
            use_pynccl=False,
        )
        self.engine = Engine(config)
        self.device = self.engine.device
        self.graphs = None
        try:
            self.trie = catalog.to(self.device)
            self.graphs = DecodeGraphs(self, max_requests) if graphs else None
        except BaseException:
            self.engine.shutdown()
            raise

    def initial_state(self, batch: int, width: int):
        return BeamState(
            torch.zeros(batch, width, device=self.device),
            torch.zeros(batch, width, dtype=torch.long, device=self.device),
            torch.zeros(batch, width, self.catalog.depth, dtype=torch.long, device=self.device),
        )

    def _decode_step(self, inputs: DecodeInput):
        logits = self.engine.model.forward().view(-1, self.width, len(self.catalog.token_ids))
        result = expand(
            logits,
            inputs.state,
            inputs.steps,
            self.trie,
            self.width,
            inputs.noise,
            fused=self.fused_search,
        )
        remap_decode_window(
            self.engine.page_table,
            inputs.rows,
            result.parents,
            inputs.prompt_lens,
            self.catalog.depth,
        )
        return result.state

    def _batch(self, reqs, input_ids, positions, out_loc, phase):
        batch = Batch(reqs, phase)
        batch.padded_reqs = reqs
        batch.input_ids = input_ids.to(device=self.device, dtype=torch.int32)
        batch.positions = positions.to(device=self.device, dtype=torch.int32)
        batch.out_loc = out_loc.to(device=self.device, dtype=torch.int32)
        self.engine.attn_backend.prepare_metadata(batch)
        return batch

    @staticmethod
    def _req(prompt, row, cached, length):
        req = Req(torch.tensor(prompt, dtype=torch.int32), row, cached, length, -1, None, None)
        return req

    def noise(self, jobs, beams):
        tensors = []
        for job in jobs:
            shape = (beams, self.catalog.max_degree)
            if job.temperature == 0:
                tensors.append(torch.zeros(shape, device=self.device))
            else:
                uniform = torch.rand(shape, generator=job.generator, device=self.device)
                tensors.append(
                    -torch.log(-torch.log(uniform.clamp(1e-7, 1 - 1e-7))) * job.temperature
                )
        return torch.stack(tensors)

    def prefill(self, jobs):
        reqs, inputs, positions, locations = [], [], [], []
        for job in jobs:
            length, cached = len(job.prompt), job.cached
            rows = job.rows
            slots = torch.tensor(job.prompt_slots, dtype=torch.int32, device=self.device)
            self.engine.page_table[rows, :length] = slots
            reqs.append(self._req(job.prompt, job.rows_cpu[0], cached, self.catalog.depth))
            inputs.extend(job.prompt[cached:])
            positions.extend(range(cached, length))
            locations.extend(job.prompt_slots[cached:])
        batch = self._batch(
            reqs, torch.tensor(inputs), torch.tensor(positions), torch.tensor(locations), "prefill"
        )
        with self.engine.ctx.forward_batch(batch):
            logits = self.engine.model.forward().view(len(jobs), 1, -1)
        state = expand(
            logits,
            self.initial_state(len(jobs), 1),
            torch.zeros(len(jobs), dtype=torch.long, device=self.device),
            self.trie,
            self.width,
            self.noise(jobs, 1),
            fused=self.fused_search,
        ).state
        for i, job in enumerate(jobs):
            job.state = BeamState(*(x[i].clone() for x in state))
            job.emitted = 1

    def decode(self, jobs):
        reqs, tokens, positions, locations = [], [], [], []
        for job in jobs:
            position = len(job.prompt) + job.emitted - 1
            writes = job.decode_slots[:, job.emitted - 1]
            self.engine.page_table[job.rows, position] = writes
            for row in job.rows_cpu:
                req = self._req([0], row, 0, self.catalog.depth)
                req.cached_len, req.device_len = position, position + 1
                reqs.append(req)
            tokens.append(job.state.tokens[:, job.emitted - 1])
            positions.extend([position] * self.width)
            locations.append(writes)
        batch = self._batch(
            reqs, torch.cat(tokens), torch.tensor(positions), torch.cat(locations), "decode"
        )
        inputs = DecodeInput(
            batch,
            BeamState(*(torch.stack([j.state[i] for j in jobs]) for i in range(3))),
            torch.tensor([j.emitted for j in jobs], device=self.device),
            torch.stack([j.rows for j in jobs]),
            torch.tensor([len(j.prompt) for j in jobs], device=self.device),
            self.noise(jobs, self.width),
        )
        if self.graphs:
            state = self.graphs.replay(inputs)
        else:
            with self.engine.ctx.forward_batch(batch):
                state = self._decode_step(inputs)
        for i, job in enumerate(jobs):
            job.state = BeamState(*(x[i].clone() for x in state))
            job.emitted += 1

    def close(self):
        if self.graphs:
            self.graphs.graphs.clear()
            self.graphs.outputs.clear()
        self.engine.shutdown()
