# Catalog-constrained recommendation

This implementation adapts the recommendation ideas in
[FlashRec](https://github.com/sohu-mptc/FlashRec) to this fork's model and FlashInfer
interfaces. Its search and scheduler are independent implementations.

The target environment remains **Python 3.12, Torch 2.9.x, ROCm 6.4,
amd-flashinfer 0.5.3+amd.2, gfx942**. Install with the ROCm instructions in the
[README](../README.md). Triton comes with the ROCm Torch distribution; do not
install a second CUDA Triton distribution over it.

## Catalog and model

Use a supported dense model trained to emit semantic item IDs (SIDs). An ordinary
chat model can exercise the runtime but does not become a useful recommender
merely by restricting its output vocabulary.

`catalog.json` is a JSON array:

```json
[
  {"item_id": "item-a", "token_ids": [1001, 2001, 3001]},
  {"item_id": "item-b", "token_ids": [1001, 2002, 3002]},
  {"item_id": "item-c", "token_ids": [1002, 2001, 3002]}
]
```

Replace these example integers with your model's actual token IDs. Every path
must have the same nonzero length, and item IDs and paths must be unique. Include
any SID boundary tokens explicitly. Catalog depth determines when generation
ends; EOS and length heuristics do not control the search.

## Serving

```bash
python -m minisgl.recommendation \
  --model /path/to/sid-model --catalog catalog.json \
  --beam-width 128 --max-requests 4 --max-pending 128 \
  --max-seq-len 4096 --memory-ratio 0.8 --port 1919
```

`minisgl-rec` is an equivalent installed entry point. There is one process with
an HTTP event loop and one thread owning the GPU engine, stream and scheduler.
Create one engine per process and initialize it before other CUDA/HIP operations.
The recommendation runtime requires TP=1 and page size=1. General mini-sglang
serving retains its own scheduler and tensor-parallel configuration.

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"input_ids":[1,42,43],"n":3,"temperature":0,"seed":7}'
```

Provide either `input_ids` or OpenAI-style text `messages`. Messages use the
model's chat template. Responses have ranked `choices`; each includes:

```json
{
  "index": 0,
  "message": {"role": "assistant", "content": "decoded SID tokens"},
  "finish_reason": "stop",
  "sglext": {
    "item_id": "item-a",
    "token_ids": [1001, 2001, 3001],
    "sequence_score": -3.25
  }
}
```

This is a non-streaming, ranked subset of the chat-completions API. `n` controls
the number returned, up to the server's beam width; fewer may survive search.
If provided, `max_tokens` must equal catalog depth. Unknown generation parameters,
including `top_p`, are rejected. `usage.completion_tokens` counts returned SID
tokens, not internal beam evaluations. `prompt_tokens_details.cached_tokens`
reports reused prompt tokens. Queue overflow returns 429, invalid requests return
400/422, and a failed worker returns 503. Disconnects cancel queued/active work at
the next scheduling boundary.

## Search semantics

The LM head projects only onto the union of catalog token IDs. Contiguous IDs use
a weight view; other IDs select the corresponding rows once at startup. Input
embeddings keep the model's full vocabulary, including when weights are tied.

For each beam, log-softmax is computed over this restricted vocabulary **before**
masking edges not allowed by the catalog trie. `sequence_score` is the sum of
these log probabilities. It is not a probability normalized over complete items,
nor the score obtained from the original full-vocabulary LM head. Comparisons
should use the same catalog vocabulary.

At temperature zero, each step keeps the highest-scoring legal expansions. This
is beam search; a finite beam can prune a path that would rank highly after later
steps. It is not exhaustive item ranking. At positive temperature, independent
Gumbel noise multiplied by temperature perturbs each step's selection scores.
Returned scores remain unperturbed and results are sorted by those scores.
This is stochastic beam search, not an exact without-replacement sampler over
complete item probabilities. Seeds use per-request generators; tie ordering and
floating-point differences can still depend on kernels/hardware.

## Execution and memory

| Component | Implementation |
| --- | --- |
| Catalog constraint | CSR trie, O(nodes + edges) persistent storage; O(requests × beams × max degree) candidate storage |
| Beam expansion | Batched device tensors, native Torch top-k, optional fused score kernel |
| KV sharing | All beams share prompt slots; forks remap only the short decode window in the page table |
| Allocation | Reserve prompt suffix plus `beam_width × (depth - 1)` KV slots at admission; each reserved decode slot is written once |
| Prefix reuse | Token trie, pinned active paths and leaf-LRU eviction; always recompute the last prompt token |
| Scheduling | Longest-prefix match plus waiting-time aging, bounded outstanding queue and beam-row capacity |
| Batching | Active requests at different SID steps share one decode batch; each retains its own prompt length, state and RNG |
| Graphs | Capture model, restricted head, search and KV-index remap for each active-group count |

The maximum decode batch has `beam_width × max_requests` rows. KV reservations
prevent a beam fork from needing an allocation midway through a request. Shared
physical KV stays immutable for the request lifetime; only page-table indices
are copied when a beam forks. Admission fails promptly if a request cannot fit
even in an empty KV pool.

Prefill runs eagerly and is batched separately from decode. This recommendation
scheduler currently uses whole prompt suffixes, without chunked prefill. Lower
`--max-requests` or the sequence limit if simultaneous long prompts use too much
temporary memory. FlashInfer planning, request admission, state-buffer copies and
RNG generation occur outside graph replay. Use `--no-graphs` for eager execution
or to diagnose capture issues. Leave memory outside the KV pool for graph and
temporary buffers; `--memory-ratio` is not an admission token limit.

## Optional precision and fusion

BF16 is the baseline. Enable optimizations independently to measure their impact:

```bash
# Fuse legal candidate scoring and Q/K normalization + RoPE + KV writes.
python -m minisgl.recommendation --model /path/to/sid-model --catalog catalog.json \
  --fused-search --fused-qk-rope

# W8A8 linears; FP8 KV is a separate choice.
python -m minisgl.recommendation --model /path/to/sid-model --catalog catalog.json \
  --quantization fp8 --kv-dtype fp8 --fused-search --fused-qk-rope
```

- W8A8 uses per-output-channel weight scales and dynamic per-token activation
  scales. Triton runs native FP8 dot products and fuses residual RMSNorm/quantize
  and SiLU/multiply/quantize. Embeddings, the output head and final norm stay BF16.
- BF16 checkpoints quantize at startup. FP8 checkpoints with scalar or per-output-
  channel `weight_scale` are dequantized before projection merging, then normalized
  to the runtime format. Scales may live in another safetensors shard. Stored
  `input_scale` values are replaced by dynamic activation scales. Block scales,
  `weight_scale_inv` and missing/invalid weight scales are rejected. Startup still
  needs space for the loaded BF16 model.
- ROCm uses E4M3 FNUZ; CUDA uses E4M3 FN. Weight values are numerically converted,
  never reinterpreted between those encodings.
- FP8 KV stores K/V with fixed scale 1 and saturating conversion. FlashInfer plans
  BF16 queries and FP8 KV separately. There is no calibration or dynamic per-token
  KV scaling; evaluate ranking quality on your model before enabling it.
- Fused search combines the log-softmax reduction, CSR gather, legality mask and
  cumulative score addition. Torch still performs top-k. Vocabularies larger than
  32768 tokens use the tensor implementation. Q/K fusion uses the model's existing
  RoPE cache, including its configured scaling.

FP8 and fusion are opt-in. Their performance advantage and ranking-quality impact
on gfx942 have not been measured here.

## Offline and profiling

```python
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.recommendation import Catalog, Recommender

config = EngineConfig(model_path="/path/to/sid-model", tp_info=DistributedInfo(0, 1),
                      dtype=torch.bfloat16, memory_ratio=0.8)
with Recommender(config, Catalog.load("catalog.json"), beam_width=128) as rec:
    results = rec.generate([[1, 42, 43], [1, 42, 44]], n=3)
    print(results[0]["beams"])
```

Start/stop traces on the GPU owner thread at step boundaries:

```bash
curl -X POST http://127.0.0.1:1919/start_profile
# Run your representative request workload.
curl -X POST http://127.0.0.1:1919/stop_profile
```

The stop response gives the Chrome/Perfetto trace path under `--profile-dir`.
Recording GPU activity requires profiler support in the installed Torch build.

## Validation

CPU tests cover exhaustive search agreement at sufficient width, illegal-path
exclusion, score semantics, KV-index forks, pinned-prefix eviction, mixed-depth
admission, cancellation, failure propagation, queue capacity, API responses,
restricted heads and FP8 scale normalization. No gfx942 device was available for
this implementation, so the new GPU kernels, graph replay and end-to-end serving
have **not** been runtime-validated there.

```bash
python -m pytest tests/recommendation tests/engine/test_rocm_config.py \
  tests/kernel/test_rocm_fallbacks.py --no-cov -q
```

GPU tests explicitly skip when no accelerator is available. On the target machine
the same command runs the kernel checks and baseline tiny-model smoke test.
Additional opt-in paths can be checked in fresh processes, without downloading
a model:

```bash
python tests/recommendation/gpu_smoke.py --fused
python tests/recommendation/gpu_smoke.py --fused --fp8
python tests/recommendation/gpu_smoke.py --fused --fp8 --fp8-kv
```

The smoke test compares eager/graph execution, repeated partial-prefix prefill
and requests at different decode steps using a tiny Qwen3 with deterministic
random weights. It checks execution parity, not trained-model accuracy. Use your
catalog and evaluation set to compare recall/NDCG, latency and throughput before
choosing FP8 or a beam width. No speedup relative to FlashRec is claimed.
