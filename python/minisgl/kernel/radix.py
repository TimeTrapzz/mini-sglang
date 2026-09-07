from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from minisgl.utils import is_rocm

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    if is_rocm():
        common_len = min(x.numel(), y.numel())
        mismatch = (x[:common_len] != y[:common_len]).nonzero()
        return common_len if mismatch.numel() == 0 else int(mismatch[0].item())
    return _load_radix_module().fast_compare_key(x, y)
