from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def _config(*, is_moe: bool = False, attention_backend: str = "auto"):
    return SimpleNamespace(
        attention_backend=attention_backend,
        model_config=SimpleNamespace(is_moe=is_moe),
        moe_backend="auto",
        page_size=1,
        use_pynccl=True,
    )


def test_rocm_selects_flashinfer_and_torch_distributed(monkeypatch):
    engine = importlib.import_module("minisgl.engine.engine")
    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    monkeypatch.setattr(engine, "is_sm90_supported", lambda: True)
    monkeypatch.setattr(engine, "is_sm100_supported", lambda: True)
    monkeypatch.setattr(engine.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine.logger, "warning_rank0", lambda *args, **kwargs: None)
    config = _config()

    engine._adjust_config(config)

    assert config.attention_backend == "fi"
    assert config.use_pynccl is False


def test_rocm_tp_uses_nccl_compatibility_backend(monkeypatch):
    engine = importlib.import_module("minisgl.engine.engine")
    config = _config()
    config.tp_info = SimpleNamespace(rank=0, size=2)
    config.distributed_timeout = 60
    config.distributed_addr = "tcp://127.0.0.1:2333"
    config.use_pynccl = False
    calls = []
    cpu_group = object()

    monkeypatch.setattr(
        engine.torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        engine.torch.distributed,
        "new_group",
        lambda **kwargs: calls.append(("new_group", kwargs)) or cpu_group,
    )
    monkeypatch.setattr(
        engine,
        "enable_pynccl_distributed",
        lambda *args, **kwargs: pytest.fail("enabled PyNCCL on ROCm"),
    )

    result = object.__new__(engine.Engine)._init_communication(config)

    assert result is cpu_group
    assert calls[0][1]["backend"] == "nccl"
    assert calls[1] == ("new_group", {"backend": "gloo"})


def test_rocm_rejects_cuda_attention_backend(monkeypatch):
    engine = importlib.import_module("minisgl.engine.engine")
    monkeypatch.setattr(engine, "is_rocm", lambda: True)

    with pytest.raises(ValueError, match="FlashInfer"):
        engine._adjust_config(_config(attention_backend="fa"))


def test_rocm_accepts_explicit_flashinfer_hybrid(monkeypatch):
    engine = importlib.import_module("minisgl.engine.engine")
    monkeypatch.setattr(engine, "is_rocm", lambda: True)
    monkeypatch.setattr(engine.logger, "warning_rank0", lambda *args, **kwargs: None)
    config = _config(attention_backend="fi,fi")

    engine._adjust_config(config)

    assert config.attention_backend == "fi,fi"


def test_rocm_rejects_moe_until_fused_ops_are_ported(monkeypatch):
    engine = importlib.import_module("minisgl.engine.engine")
    monkeypatch.setattr(engine, "is_rocm", lambda: True)

    with pytest.raises(NotImplementedError, match="dense models only"):
        engine._adjust_config(_config(is_moe=True))
