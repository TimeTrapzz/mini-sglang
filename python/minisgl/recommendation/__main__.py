from __future__ import annotations

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="Catalog-constrained SID recommendation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", required=True, help="JSON item_id / token_ids catalog")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-requests", type=int, default=4)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--memory-ratio", type=float, default=0.8)
    parser.add_argument("--num-pages", type=int)
    parser.add_argument("--quantization", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--fused-qk-rope", action="store_true")
    parser.add_argument("--fused-search", action="store_true")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--profile-dir", default="profiles")
    args = parser.parse_args()
    if args.beam_width < 1 or args.max_requests < 1 or not 0 < args.memory_ratio < 1:
        parser.error("beam width / max requests must be positive; memory ratio must be in (0, 1)")
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig
    from minisgl.utils import load_tokenizer

    from .catalog import Catalog
    from .precision import fp8_dtype
    from .runtime import RecommendationRuntime
    from .scheduler import RecommendationWorker
    from .server import create_app

    catalog = Catalog.load(args.catalog)
    tokenizer = load_tokenizer(args.model)
    config = EngineConfig(
        model_path=args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_seq_len_override=args.max_seq_len,
        memory_ratio=args.memory_ratio,
        num_page_override=args.num_pages,
        quantization="fp8" if args.quantization == "fp8" else None,
        kv_cache_dtype=fp8_dtype() if args.kv_dtype == "fp8" else None,
        fused_qk_rope=args.fused_qk_rope,
    )
    worker = RecommendationWorker(
        lambda: RecommendationRuntime(
            config,
            catalog,
            args.beam_width,
            args.max_requests,
            graphs=not args.no_graphs,
            fused_search=args.fused_search,
        ),
        args.max_requests,
        args.max_pending,
        profile_dir=args.profile_dir,
    )
    import uvicorn

    try:
        uvicorn.run(create_app(worker, tokenizer, args.model), host=args.host, port=args.port)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
