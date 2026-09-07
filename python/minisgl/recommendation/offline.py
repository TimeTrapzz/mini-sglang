from __future__ import annotations

from itertools import islice

from .catalog import Catalog
from .scheduler import RecommendationWorker


class Recommender:
    """Offline facade. Construct before initializing CUDA; one engine per process."""

    def __init__(
        self,
        config,
        catalog: Catalog,
        *,
        beam_width=128,
        max_requests=4,
        max_pending=128,
        graphs=True,
        fused_search=False,
    ):
        from .runtime import RecommendationRuntime

        self.worker = RecommendationWorker(
            lambda: RecommendationRuntime(
                config, catalog, beam_width, max_requests, graphs=graphs, fused_search=fused_search
            ),
            max_requests=max_requests,
            max_pending=max_pending,
        )

    def generate(self, prompts, *, n=1, temperature=0.0, seed=0):
        """Return ranked items in prompt order, submitting bounded batches."""
        source, results = iter(prompts), []
        while chunk := list(islice(source, self.worker.max_pending)):
            futures = []
            try:
                for prompt in chunk:
                    futures.append(self.worker.submit(prompt, n, temperature, seed))
                results.extend(future.result() for future in futures)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results

    def close(self):
        self.worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
