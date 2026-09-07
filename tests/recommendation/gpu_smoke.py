"""Run in a fresh process: tiny Qwen3, eager/graph, cached prefill and mixed depths.

Uses deterministic random weights and needs no model download. This verifies
execution parity, not recommendation quality or quantized-model accuracy.
"""

import argparse
import tempfile

import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.recommendation import Catalog
from minisgl.recommendation.precision import fp8_dtype
from minisgl.recommendation.runtime import RecommendationRuntime
from minisgl.recommendation.scheduler import Job
from transformers import Qwen3Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-kv", action="store_true")
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder, torch.inference_mode():
        hf = Qwen3Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        hf.architectures = ["Qwen3ForCausalLM"]
        hf.save_pretrained(folder)
        config = EngineConfig(
            model_path=folder,
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            use_dummy_weight=True,
            num_page_override=512,
            quantization="fp8" if args.fp8 else None,
            kv_cache_dtype=fp8_dtype() if args.fp8_kv else None,
            fused_qk_rope=args.fused,
        )
        catalog = Catalog(
            [
                {"item_id": str(p), "token_ids": list(p)}
                for p in [(10, 20, 30), (10, 21, 31), (11, 20, 31), (11, 21, 30)]
            ]
        )
        runtime = RecommendationRuntime(config, catalog, 4, 2, fused_search=args.fused)
        captured = runtime.graphs
        next_slot = 0

        def job(prompt, group, previous=None):
            nonlocal next_slot
            result = Job(prompt, 4)
            result.cached = len(prompt) - 1 if previous else 0
            count = len(prompt) - result.cached + 4 * (catalog.depth - 1)
            own = list(range(next_slot, next_slot + count))
            next_slot += count
            prefix = previous.prompt_slots[: result.cached] if previous else []
            split = len(prompt) - result.cached
            result.prompt_slots = prefix + own[:split]
            result.decode_slots = torch.tensor(own[split:], device="cuda", dtype=torch.int32).view(
                4, -1
            )
            result.rows_cpu = list(range(group * 4, (group + 1) * 4))
            result.rows = torch.tensor(result.rows_cpu, device="cuda")
            return result

        def run(graphs, previous=None, mixed=False):
            runtime.graphs = captured if graphs else None
            jobs = [
                job([1, 2, 3], 0, previous[0] if previous else None),
                job([1, 2, 4, 5], 1, previous[1] if previous else None),
            ]
            if mixed:
                runtime.prefill(jobs[:1])
                runtime.decode(jobs[:1])
                runtime.prefill(jobs[1:])
            else:
                runtime.prefill(jobs)
            while active := [j for j in jobs if j.emitted < catalog.depth]:
                runtime.decode(active)
            return jobs

        try:
            baseline = run(False)
            for graphs, cached, mixed in [
                (True, False, False),
                (False, True, False),
                (True, True, False),
                (True, True, True),
            ]:
                result = run(graphs, baseline if cached else None, mixed)
                for actual, expected in zip(result, baseline):
                    assert torch.isfinite(actual.state.scores).all()
                    torch.testing.assert_close(actual.state.tokens, expected.state.tokens)
                    torch.testing.assert_close(
                        actual.state.scores, expected.state.scores, rtol=0.01, atol=0.03
                    )
            print("PASS: eager/graph, partial-prefix prefill and mixed-depth batching")
        finally:
            runtime.graphs = captured
            runtime.close()


if __name__ == "__main__":
    main()
