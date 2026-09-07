from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from minisgl.utils import is_rocm

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    if is_rocm():
        if k_cache.element_size() == 1:
            import torch

            # index_copy does not support every FP8 dtype. Copy the existing
            # representation byte-for-byte; conversion happens in the KV pool.
            k_cache, v_cache, k, v = (x.view(torch.uint8) for x in (k_cache, v_cache, k, v))
        indices = indices.long()
        k_cache.index_copy_(0, indices, k.view(k.shape[0], -1))
        v_cache.index_copy_(0, indices, v.view(v.shape[0], -1))
        return

    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
