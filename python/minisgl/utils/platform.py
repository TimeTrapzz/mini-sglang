from __future__ import annotations


def is_rocm() -> bool:
    import torch

    return torch.version.hip is not None
