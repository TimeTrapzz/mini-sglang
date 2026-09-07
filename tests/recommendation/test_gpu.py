"""Numerical kernel checks: skipped explicitly when no GPU is present."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.recommendation import BeamState, Catalog, expand
from minisgl.recommendation.precision import FP8Linear, FP8RMSNorm, fp8_dtype, quantize_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/ROCm GPU")


def test_tiny_model_graph_and_cache_parity():
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("gpu_smoke.py"))], check=True, timeout=240
    )


def test_fused_search_matches_tensor_reference():
    catalog = Catalog(
        [{"item_id": str(p), "token_ids": list(p)} for p in [(1, 2, 3), (1, 4, 3), (4, 2, 4)]]
    )
    trie = catalog.to("cuda")
    state = BeamState(
        torch.zeros(2, 1, device="cuda"),
        torch.zeros(2, 1, dtype=torch.long, device="cuda"),
        torch.zeros(2, 1, 3, dtype=torch.long, device="cuda"),
    )
    torch.manual_seed(11)
    for step in range(3):
        logits = torch.randn(2, state.scores.shape[1], len(catalog.token_ids), device="cuda")
        steps = torch.full((2,), step, device="cuda", dtype=torch.long)
        expected = expand(logits, state, steps, trie, 4)
        result = expand(logits, state, steps, trie, 4, fused=True)
        live = torch.isfinite(expected.state.scores)
        torch.testing.assert_close(result.state.scores, expected.state.scores)
        torch.testing.assert_close(result.state.tokens[live], expected.state.tokens[live])
        state = expected.state


def test_fp8_gemm_matches_explicit_dequantization():
    torch.manual_seed(12)
    x = torch.randn(17, 96, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(71, 96, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(71, device="cuda", dtype=torch.bfloat16)
    linear = FP8Linear(SimpleNamespace(weight=w, bias=bias))
    qx, qw = quantize_rows(x), linear._weights
    reference = (qx.values.float() * qx.scales) @ (qw.values.float() * qw.scales).T + bias.float()
    torch.testing.assert_close(linear.forward(qx).float(), reference, rtol=0.01, atol=0.125)


@pytest.mark.parametrize("residual", [False, True])
def test_fused_rms_and_silu_quantization(residual):
    from minisgl.recommendation.precision import silu_quantize

    torch.manual_seed(13)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x) if residual else None
    w = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    norm = FP8RMSNorm(SimpleNamespace(weight=w, eps=1e-6))
    result, res = norm.forward(x, r)
    expected_res = x if r is None else x + r
    expected = expected_res.float()
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(res, expected_res)
    torch.testing.assert_close(result.values.float() * result.scales, expected, rtol=0.1, atol=0.15)
    act = silu_quantize(x)
    gate, up = x.float().chunk(2, dim=-1)
    torch.testing.assert_close(
        act.values.float() * act.scales, torch.nn.functional.silu(gate) * up, rtol=0.1, atol=0.15
    )


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
def test_qk_rope_and_kv_store(norm, cache_dtype):
    from minisgl.recommendation.triton_ops import qk_rope_store

    torch.manual_seed(14)
    t, nq, nk, d = 3, 4, 1, 64
    qkv = torch.randn(t, (nq + 2 * nk) * d, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([1, 4, 7], device="cuda", dtype=torch.int32)
    locations = torch.tensor([2, 8, 5], device="cuda", dtype=torch.int32)
    angles = torch.randn(10, d // 2, device="cuda")
    cos_sin = torch.cat([angles.cos(), angles.sin()], -1)
    qnorm = SimpleNamespace(weight=torch.randn(d, device="cuda", dtype=torch.bfloat16), eps=1e-6)
    knorm = SimpleNamespace(weight=torch.randn(d, device="cuda", dtype=torch.bfloat16), eps=1e-6)
    dtype = fp8_dtype() if cache_dtype == "fp8" else torch.bfloat16
    kc = torch.zeros(10, 1, nk, d, device="cuda", dtype=dtype)
    vc = torch.zeros_like(kc)
    layer = SimpleNamespace(
        num_qo_heads=nq,
        num_kv_heads=nk,
        head_dim=d,
        layer_id=0,
        q_norm=qnorm if norm else None,
        k_norm=knorm if norm else None,
        rotary=SimpleNamespace(_cos_sin_cache=cos_sin),
    )
    cache = SimpleNamespace(dtype=dtype, k_cache=lambda _: kc, v_cache=lambda _: vc)
    ctx = SimpleNamespace(
        kv_cache=cache, batch=SimpleNamespace(positions=positions, out_loc=locations)
    )
    actual_q = qk_rope_store(qkv, layer, ctx)
    q, k, v = qkv.split([nq * d, nk * d, nk * d], -1)

    def reference(x, heads, weight):
        x = x.view(t, heads, d).float()
        if norm:
            x = (
                (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight)
                .bfloat16()
                .float()
            )
        c = angles[positions.long()].cos().repeat(1, 2)[:, None]
        s = angles[positions.long()].sin().repeat(1, 2)[:, None]
        x = x * c + torch.cat([-x[..., d // 2 :], x[..., : d // 2]], -1) * s
        return x.bfloat16()

    expected_q, expected_k = reference(q, nq, qnorm.weight), reference(k, nk, knorm.weight)
    torch.testing.assert_close(actual_q, expected_q, rtol=0.02, atol=0.04)
    tolerance = 0.15 if cache_dtype == "fp8" else 0.04
    torch.testing.assert_close(
        kc.float()[locations.long(), 0], expected_k.float(), rtol=0.08, atol=tolerance
    )
    torch.testing.assert_close(
        vc.float()[locations.long(), 0], v.view(t, nk, d).float(), rtol=0.08, atol=tolerance
    )
    assert kc.float()[0].count_nonzero() == 0
