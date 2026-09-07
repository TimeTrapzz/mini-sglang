import queue
import threading
from types import SimpleNamespace

import pytest
import torch
from minisgl.recommendation import BeamState, Catalog, expand
from minisgl.recommendation.scheduler import RecommendationWorker


class FakeRuntime:
    """A deterministic model; the real search, admission and cache run unchanged."""

    def __init__(self, *, blocked=False, fail=False, depth=4, capacity=64):
        self.device, self.width, self.vocab_size = torch.device("cpu"), 2, 100
        self.engine = SimpleNamespace(num_pages=capacity, max_seq_len=32)
        self.catalog = Catalog(
            [
                {"item_id": "a", "token_ids": [10] * depth},
                {"item_id": "b", "token_ids": [20] * depth},
            ]
        )
        self.trie = self.catalog.to("cpu")
        self.entered, self.resume = threading.Event(), threading.Event()
        if not blocked:
            self.resume.set()
        self.fail, self.closed, self.decode_batches = fail, False, []

    def prefill(self, jobs):
        self.entered.set()
        assert self.resume.wait(5)
        if self.fail:
            raise RuntimeError("synthetic GPU failure")
        for job in jobs:
            state = BeamState(
                torch.zeros(1, 1),
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1, self.catalog.depth, dtype=torch.long),
            )
            result = expand(torch.tensor([[[2.0, 1.0]]]), state, torch.tensor([0]), self.trie, 2)
            job.state = BeamState(*(x[0] for x in result.state))
            job.emitted = 1

    def decode(self, jobs):
        self.decode_batches.append([j.emitted for j in jobs])
        for job in jobs:
            state = BeamState(*(x[None] for x in job.state))
            result = expand(
                torch.tensor([[[2.0, 1.0], [2.0, 1.0]]]),
                state,
                torch.tensor([job.emitted]),
                self.trie,
                2,
            )
            job.state = BeamState(*(x[0] for x in result.state))
            job.emitted += 1

    def close(self):
        self.closed = True
        self.resume.set()


def test_mixed_depth_batching_and_cached_repeat():
    runtime = FakeRuntime(blocked=True)
    worker = RecommendationWorker(lambda: runtime, max_requests=3)
    try:
        a = worker.submit([1, 2, 3], 2)
        assert runtime.entered.wait(2)
        b = worker.submit([1, 2, 4], 2)
        c = worker.submit([9, 8], 1)
        runtime.resume.set()
        results = [f.result(5) for f in (a, b, c)]
        assert any(len(batch) == 3 and len(set(batch)) > 1 for batch in runtime.decode_batches)
        assert [r["beams"][0]["item_id"] for r in results] == ["a"] * 3
        assert results[1]["cached_tokens"] == 2
        repeated = worker.submit([1, 2, 3], 2).result(5)
        assert repeated["cached_tokens"] == 2  # Last prompt token is always recomputed.
        assert repeated["beams"] == results[0]["beams"]
    finally:
        runtime.resume.set()
        worker.close()
    assert runtime.closed


def test_bounded_queue_cancel_and_release():
    runtime = FakeRuntime(blocked=True)
    worker = RecommendationWorker(lambda: runtime, max_pending=2)
    try:
        first = worker.submit([1], 1)
        assert runtime.entered.wait(2)
        cancelled = worker.submit([2], 1)
        with pytest.raises(queue.Full):
            worker.submit([3], 1)
        assert cancelled.cancel()
        replacement = worker.submit([4], 1)
        runtime.resume.set()
        assert first.result(5)["beams"]
        assert replacement.result(5)["beams"]
    finally:
        runtime.resume.set()
        worker.close()


def test_worker_failure_reaches_active_pending_and_future_requests():
    runtime = FakeRuntime(blocked=True, fail=True)
    worker = RecommendationWorker(lambda: runtime)
    first = worker.submit([1], 1)
    assert runtime.entered.wait(2)
    pending = worker.submit([2], 1)
    runtime.resume.set()
    for future in (first, pending):
        with pytest.raises(RuntimeError, match="synthetic GPU failure"):
            future.result(5)
    with pytest.raises(RuntimeError, match="unavailable"):
        worker.submit([3], 1)
    worker.close()
    assert runtime.closed


def test_startup_failure_is_propagated():
    def fail():
        raise ValueError("bad checkpoint")

    with pytest.raises(RuntimeError, match="startup failed") as error:
        RecommendationWorker(fail)
    assert isinstance(error.value.__cause__, ValueError)


def test_oversized_request_fails_without_stalling_queue():
    runtime = FakeRuntime(capacity=8)
    worker = RecommendationWorker(lambda: runtime)
    try:
        with pytest.raises(ValueError, match="KV capacity"):
            worker.submit([1, 2, 3], 1).result(5)
        assert worker.submit([1], 1).result(5)["beams"]
    finally:
        worker.close()


def test_depth_one_and_input_validation():
    runtime = FakeRuntime(depth=1)
    worker = RecommendationWorker(lambda: runtime)
    try:
        assert len(worker.submit([1], 2).result(5)["beams"]) == 2
        for prompt, n, temp, seed in [
            ([], 1, 0, 0),
            ([100], 1, 0, 0),
            ([1], 3, 0, 0),
            ([1], 1, float("nan"), 0),
            ([1], 1, 0, -1),
        ]:
            with pytest.raises(ValueError):
                worker.submit(prompt, n, temp, seed)
    finally:
        worker.close()
