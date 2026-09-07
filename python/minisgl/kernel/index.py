from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from minisgl.utils import is_rocm

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


def _indexing_torch(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    vocab_range: Tuple[int, int] | None,
) -> torch.Tensor:
    if vocab_range is None:
        output.copy_(weights.index_select(0, indices.long()))
        return output

    start, length = vocab_range
    local_indices = indices - start
    valid = (local_indices >= 0) & (local_indices < length)
    local_indices = local_indices.masked_fill(~valid, 0)
    output.copy_(weights.index_select(0, local_indices.long()))
    output.masked_fill_(~valid.unsqueeze(1), 0)
    return output


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    if is_rocm():
        return _indexing_torch(weights, indices, output, vocab_range)

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
