from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.recommendation.checkpoint import checkpoint_tensors
from minisgl.recommendation.precision import FP8Linear, FP8RMSNorm, quantize_rows, silu_quantize
from safetensors.torch import save_file


def dequant(x):
    return x.values.float() * x.scales


def test_fp8_per_channel_and_per_token_scales_with_zero_rows():
    torch.manual_seed(8)
    x = torch.randn(7, 64).bfloat16()
    x[0].zero_()
    weight = torch.randn(33, 64).bfloat16()
    weight[0].zero_()
    weight[1].mul_(100)
    bias = torch.randn(33).bfloat16()
    output = FP8Linear(SimpleNamespace(weight=weight, bias=bias)).forward(x).float()
    expected = F.linear(x.float(), weight.float(), bias.float())
    # Different channel magnitudes must not poison the small channels.
    error = (output - expected).square().mean(0).sqrt()
    reference = expected.square().mean(0).sqrt().clamp_min(1.0)
    assert (error / reference).max() < 0.12
    assert torch.isfinite(dequant(quantize_rows(x))).all()


def test_fused_precision_reference_residual_and_activation():
    torch.manual_seed(9)
    x, residual = torch.randn(5, 64).bfloat16(), torch.randn(5, 64).bfloat16()
    weight = torch.randn(64).bfloat16()
    norm = FP8RMSNorm(SimpleNamespace(weight=weight, eps=1e-6))
    quant, new_residual = norm.forward(x, residual)
    expected = (x + residual).float()
    expected *= torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
    expected *= weight.float()
    torch.testing.assert_close(new_residual, x + residual)
    torch.testing.assert_close(dequant(quant), expected, atol=0.16, rtol=0.08)
    gate, up = x.chunk(2, dim=-1)
    torch.testing.assert_close(
        dequant(silu_quantize(x)), (F.silu(gate) * up).float(), atol=0.1, rtol=0.08
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
@pytest.mark.parametrize("scale", [torch.tensor(0.5), torch.tensor([0.5, 2.0])])
def test_checkpoint_scales_can_live_in_another_shard(tmp_path, dtype, scale):
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(dtype)
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"proj.weight": weight}, str(a))
    save_file({"proj.weight_scale": scale, "proj.input_scale": torch.tensor(99.0)}, str(b))
    result = dict(checkpoint_tensors([a, b], "cpu"))
    assert set(result) == {"proj.weight"}
    torch.testing.assert_close(result["proj.weight"].float(), weight.float() * scale.reshape(-1, 1))


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"proj.weight_scale": torch.ones(2, 2)},
        {"proj.weight_scale": torch.tensor(0.0)},
        {"proj.weight_scale_inv": torch.tensor(1.0)},
    ],
)
def test_checkpoint_rejects_missing_ambiguous_or_invalid_scales(tmp_path, metadata):
    path = tmp_path / "model.safetensors"
    save_file({"proj.weight": torch.ones(2, 2).to(torch.float8_e4m3fn), **metadata}, str(path))
    with pytest.raises(ValueError):
        dict(checkpoint_tensors([path], "cpu"))


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("ids", [[2, 3, 4], [7, 1, 5]])
def test_restricted_head_matches_full_projection(monkeypatch, tied, ids):
    from minisgl.layers import embedding

    head = object.__new__(embedding.ParallelLMHead)
    head.tp_size, head.num_embeddings = 1, 10
    head.weight, head.bias = torch.randn(10, 16), torch.randn(10)
    head.tied_embedding = SimpleNamespace(weight=torch.randn(10, 16)) if tied else None
    x = torch.randn(5, 16)
    selected_rows = torch.tensor([1, 4])
    batch = SimpleNamespace(
        size=2,
        is_prefill=True,
        attn_metadata=SimpleNamespace(get_last_indices=lambda _: selected_rows),
    )
    monkeypatch.setattr(embedding, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    forward = getattr(
        embedding.ParallelLMHead.forward, "__wrapped__", embedding.ParallelLMHead.forward
    )
    full = forward(head, x)
    head.restrict(ids)
    torch.testing.assert_close(forward(head, x), full[:, ids])


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_fp8_kv_conversion_and_rocm_byte_copy(monkeypatch, dtype):
    import importlib

    pool_module = importlib.import_module("minisgl.kvcache.mha_pool")
    store = importlib.import_module("minisgl.kernel.store")
    monkeypatch.setattr(pool_module, "get_tp_info", lambda: SimpleNamespace(size=1))
    monkeypatch.setattr(store, "is_rocm", lambda: True)
    cache = pool_module.MHAKVCache(1, 1, 4, 8, 1, dtype, torch.device("cpu"))
    k = torch.tensor([[1.0, -2.0, 700.0, -700.0], [0.0, 0.1, 0.5, 1.0]]).bfloat16()
    v = -k
    slots = torch.tensor([2, 5], dtype=torch.int32)
    cache.store_kv(k, v, slots, 0)
    limit = torch.finfo(dtype).max
    for stored, source in [(cache.k_cache(0), k), (cache.v_cache(0), v)]:
        expected = source.float().clamp(-limit, limit).to(dtype).view(torch.uint8)
        actual = stored.view(8, 4).view(torch.uint8)[slots.long()]
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("fp8", [False, True])
def test_weight_loader_merges_projections_after_scale_conversion(monkeypatch, tmp_path, fp8):
    import importlib

    loader = importlib.import_module("minisgl.models.weight")
    config = importlib.import_module("minisgl.models.config")
    monkeypatch.setattr(loader, "download_hf_weight", lambda _: str(tmp_path))
    monkeypatch.setattr(loader, "cached_load_hf_config", lambda _: None)
    monkeypatch.setattr(
        loader, "get_tp_info", lambda: SimpleNamespace(rank=0, size=1, is_primary=lambda: True)
    )
    monkeypatch.setattr(
        config.ModelConfig, "from_hf", lambda _: SimpleNamespace(num_kv_heads=1, is_moe=False)
    )
    weights, scales, expected = {}, {}, []
    for i, projection in enumerate(("q", "k", "v"), 1):
        key = f"model.layers.0.self_attn.{projection}_proj"
        value = torch.full((2, 4), float(i), dtype=torch.bfloat16)
        weights[key + ".weight"] = value.to(torch.float8_e4m3fn) if fp8 else value
        if fp8:
            scales[key + ".weight_scale"] = torch.tensor([0.5, 2.0])
            value = value * torch.tensor([[0.5], [2.0]])
        expected.append(value.bfloat16())
    save_file(weights, str(tmp_path / "weights.safetensors"))
    if scales:
        save_file(scales, str(tmp_path / "scales.safetensors"))
    result = dict(loader.load_weight("local", torch.device("cpu"), dequantize_fp8=fp8))
    assert set(result) == {"model.layers.0.self_attn.qkv_proj.weight"}
    torch.testing.assert_close(next(iter(result.values())), torch.cat(expected))
