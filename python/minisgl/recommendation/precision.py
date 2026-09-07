from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import _LinearTPImpl


class QuantizedRows(NamedTuple):
    values: torch.Tensor
    scales: torch.Tensor
    output_dtype: torch.dtype


def fp8_dtype():
    # CDNA3 uses FNUZ encodings. Never reinterpret FN storage as FNUZ.
    return torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn


def quantize_rows(x: torch.Tensor) -> QuantizedRows:
    if x.is_cuda:
        from .triton_ops import quantize

        return quantize(x)
    limit = torch.finfo(fp8_dtype()).max
    scales = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / limit
    values = (x.float() / scales).clamp(-limit, limit).to(fp8_dtype())
    return QuantizedRows(values, scales, x.dtype)


class FP8Linear(BaseOP):
    """W8A8 with per-output-channel weights and dynamic per-token activations."""

    def __init__(self, source):
        self._weights = quantize_rows(source.weight)
        self.bias = source.bias

    def forward(self, x):
        x = x if isinstance(x, QuantizedRows) else quantize_rows(x)
        if x.values.is_cuda:
            from .triton_ops import matmul

            return matmul(x, self._weights, self.bias)
        a = x.values.float() * x.scales
        b = self._weights.values.float() * self._weights.scales
        return F.linear(a, b, None if self.bias is None else self.bias.float()).to(x.output_dtype)


class FP8RMSNorm(BaseOP):
    def __init__(self, source):
        self.weight, self.eps = source.weight, source.eps

    def forward(self, x, residual=None):
        if x.is_cuda:
            from .triton_ops import rms_quantize

            return rms_quantize(x, residual, self.weight, self.eps)
        residual = x if residual is None else x + residual
        normalized = residual.float() * torch.rsqrt(
            residual.float().square().mean(-1, keepdim=True) + self.eps
        )
        normalized = (normalized * self.weight.float()).to(x.dtype)
        return quantize_rows(normalized), residual


def silu_quantize(x):
    if x.is_cuda:
        from .triton_ops import silu_quantize as fused

        return fused(x)
    gate, up = x.chunk(2, dim=-1)
    return quantize_rows(F.silu(gate) * up)


def quantize_model(model, mode: str):
    if mode != "fp8":
        raise ValueError("Supported quantization: fp8")
    from minisgl.distributed import get_tp_info
    from minisgl.layers import silu_and_mul
    from minisgl.models.utils import GatedMLP

    if get_tp_info().size != 1:
        raise ValueError("FP8 recommendation kernels currently require one GPU")

    def visit(op):
        for name, child in list(vars(op).items()):
            if isinstance(child, _LinearTPImpl):
                setattr(op, name, FP8Linear(child))
            elif name in ("input_layernorm", "post_attention_layernorm"):
                setattr(op, name, FP8RMSNorm(child))
            elif isinstance(child, OPList):
                for item in child.op_list:
                    visit(item)
            elif isinstance(child, BaseOP):
                visit(child)
        if isinstance(op, GatedMLP) and op.act_fn is silu_and_mul:
            op.act_fn = silu_quantize

    visit(model)
