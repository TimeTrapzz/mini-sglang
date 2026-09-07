from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .cache import PrefixPool


@dataclass
class Job:
    prompt: list[int]
    n: int
    temperature: float = 0.0
    seed: int = 0
    future: Future = field(default_factory=Future)
    arrived: float = field(default_factory=time.monotonic)
    group: int = -1
    emitted: int = 0
    cached: int = 0
    owned: list = field(default_factory=list)
    pins: list = field(default_factory=list)
    adopted: set = field(default_factory=set)
    state: object = None


def complete(future, *, result=None, error=None):
    try:
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)
    except InvalidStateError:  # An HTTP client may cancel during the last GPU step.
        pass


class RecommendationWorker:
    """One GPU owner thread; HTTP/tokenization share its process, not its stream.

    runtime_factory is injectable so admission, failure and cancellation are CPU-testable.
    """

    def __init__(
        self,
        runtime_factory,
        max_requests: int = 4,
        max_pending: int = 128,
        aging_seconds: float = 0.1,
        profile_dir: str = "profiles",
    ):
        if min(max_requests, max_pending) < 1 or aging_seconds <= 0:
            raise ValueError("Request limits and aging interval must be positive")
        self.factory, self.max_requests = runtime_factory, max_requests
        self.aging_seconds, self.max_pending = aging_seconds, max_pending
        self.inbox = queue.Queue(max_pending)
        self.permits = threading.BoundedSemaphore(max_pending)
        self.ready, self.stopping = threading.Event(), threading.Event()
        self.submit_lock = threading.Lock()
        self.controls = queue.Queue()
        self.profiler, self.profile_dir = None, Path(profile_dir)
        self.error = None
        self.thread = threading.Thread(target=self._run, name="minisgl-recommendation", daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise RuntimeError("Recommendation runtime startup failed") from self.error

    def submit(self, prompt, n, temperature=0.0, seed=0) -> Future:
        if self.stopping.is_set() or self.error:
            raise RuntimeError("Recommendation worker is unavailable")
        if not prompt or any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError("prompt token IDs must be nonnegative integers")
        if max(prompt) >= self.runtime.vocab_size:
            raise ValueError("Prompt token ID exceeds the model vocabulary")
        if type(n) is not int or not 1 <= n <= self.runtime.width:
            raise ValueError(f"n must be between 1 and configured beam width {self.runtime.width}")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63 - 1]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if len(prompt) + self.runtime.catalog.depth > self.runtime.engine.max_seq_len:
            raise ValueError("Prompt plus SID exceeds max sequence length")
        if not self.permits.acquire(blocking=False):
            raise queue.Full("Recommendation queue is full")
        job = Job(list(prompt), n, temperature, seed)
        job.future.add_done_callback(lambda _: self.permits.release())
        with self.submit_lock:
            if self.stopping.is_set():
                job.future.cancel()
                raise RuntimeError("Recommendation worker stopped")
            try:
                self.inbox.put_nowait(job)
            except queue.Full:
                job.future.cancel()
                raise
        return job.future

    def profile(self, action):
        future = Future()
        with self.submit_lock:
            if self.stopping.is_set():
                raise RuntimeError("Recommendation worker stopped")
            self.controls.put((action, future))
        return future

    def _profile(self):
        while not self.controls.empty():
            action, future = self.controls.get_nowait()
            try:
                if action == "start":
                    if self.profiler is not None:
                        raise RuntimeError("Profiler is already active")
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if self.runtime.device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    self.profiler = torch.profiler.profile(
                        activities=activities, record_shapes=True
                    )
                    self.profiler.__enter__()
                    result = {"status": "started"}
                elif action == "stop" and self.profiler is not None:
                    profiler, self.profiler = self.profiler, None
                    profiler.__exit__(None, None, None)
                    self.profile_dir.mkdir(parents=True, exist_ok=True)
                    path = self.profile_dir / f"recommendation-{time.time_ns()}.json"
                    profiler.export_chrome_trace(str(path))
                    result = {"status": "stopped", "trace": str(path)}
                else:
                    raise RuntimeError("Profiler is not active")
                complete(future, result=result)
            except Exception as exc:
                complete(future, error=exc)

    def _admit(self, pending, free_groups):
        admitted = []
        now = time.monotonic()
        pending.sort(
            key=lambda j: (
                len(self.pool.match(j.prompt[:-1])) + (now - j.arrived) / self.aging_seconds
            ),
            reverse=True,
        )
        for job in list(pending):
            if job.future.cancelled():
                pending.remove(job)
                continue
            if not free_groups:
                break
            pins = self.pool.match(job.prompt[:-1], pin=True)
            needed = (
                len(job.prompt) - len(pins) + self.runtime.width * (self.runtime.catalog.depth - 1)
            )
            if needed > self.pool.available:
                self.pool.release(pins)
                # Requests too large even in an empty pool must fail, not wait forever.
                total = len(job.prompt) + self.runtime.width * (self.runtime.catalog.depth - 1)
                if total > self.pool.capacity:
                    pending.remove(job)
                    complete(job.future, error=ValueError("Request exceeds KV capacity"))
                continue
            job.pins, job.cached = pins, len(pins)
            job.owned = self.pool.allocate(needed)
            job.group = free_groups.pop()
            job.rows_cpu = list(
                range(job.group * self.runtime.width, (job.group + 1) * self.runtime.width)
            )
            job.rows = torch.tensor(job.rows_cpu, device=self.runtime.device)
            suffix = len(job.prompt) - job.cached
            job.prompt_slots = [n.slot for n in pins] + job.owned[:suffix]
            job.decode_slots = torch.tensor(
                job.owned[suffix:], dtype=torch.int32, device=self.runtime.device
            ).view(self.runtime.width, -1)
            job.generator = torch.Generator(device=self.runtime.device).manual_seed(job.seed)
            admitted.append(job)
            pending.remove(job)
        return admitted

    def _free(self, job, free_groups):
        self.pool.release(job.pins)
        self.pool.free_owned(job.owned, job.adopted)
        free_groups.append(job.group)

    def _finish(self, job):
        scores = job.state.scores.float().cpu().tolist()
        if any(math.isnan(score) or score == math.inf for score in scores):
            raise RuntimeError("Model produced non-finite recommendation scores")
        tokens = job.state.tokens.cpu().tolist()
        beams = []
        for score, path in sorted(zip(scores, tokens), key=lambda pair: -pair[0]):
            if math.isfinite(score):
                beams.append(
                    {
                        "item_id": self.runtime.catalog.items[tuple(path)],
                        "token_ids": path,
                        "sequence_score": score,
                    }
                )
            if len(beams) == job.n:
                break
        complete(
            job.future,
            result={
                "beams": beams,
                "prompt_tokens": len(job.prompt),
                "cached_tokens": job.cached,
                "completion_tokens": len(beams) * self.runtime.catalog.depth,
            },
        )

    def _run(self):
        pending, active = [], []
        self.runtime = None
        try:
            with torch.inference_mode():
                self.runtime = self.factory()
                self.pool = PrefixPool(self.runtime.engine.num_pages)
                free_groups = list(range(self.max_requests))
                self.ready.set()
                while not self.stopping.is_set():
                    try:
                        pending.append(self.inbox.get(timeout=0.02 if not active else 0))
                    except queue.Empty:
                        pass
                    while not self.inbox.empty():
                        pending.append(self.inbox.get_nowait())
                    self._profile()
                    for job in list(active):
                        if job.future.cancelled():
                            self._free(job, free_groups)
                            active.remove(job)
                    admitted = self._admit(pending, free_groups)
                    active.extend(admitted)
                    if admitted:
                        self.runtime.prefill(admitted)
                        for job in admitted:
                            pins, job.adopted = self.pool.insert(job.prompt, job.prompt_slots)
                            self.pool.release(job.pins)
                            job.pins = pins
                    decoding = [j for j in active if j.emitted < self.runtime.catalog.depth]
                    if decoding:
                        self.runtime.decode(decoding)
                    for job in list(active):
                        if job.emitted == self.runtime.catalog.depth:
                            self._finish(job)
                            self._free(job, free_groups)
                            active.remove(job)
        except BaseException as exc:
            self.error = exc
        finally:
            with self.submit_lock:
                self.stopping.set()
            self.ready.set()
            while not self.inbox.empty():
                pending.append(self.inbox.get_nowait())
            error = self.error or RuntimeError("Recommendation worker stopped")
            for job in pending + active:
                complete(job.future, error=error)
            while not self.controls.empty():
                _, future = self.controls.get_nowait()
                complete(future, error=error)
            if self.profiler is not None:
                self.profiler.__exit__(None, None, None)
            if self.runtime is not None:
                self.runtime.close()

    def close(self):
        with self.submit_lock:
            self.stopping.set()
        self.thread.join()
