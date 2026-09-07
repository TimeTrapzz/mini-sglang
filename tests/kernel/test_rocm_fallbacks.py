from __future__ import annotations

import importlib

import pytest
import torch


def test_rocm_detection(monkeypatch):
    platform = importlib.import_module("minisgl.utils.platform")
    monkeypatch.setattr(torch.version, "hip", "7.2")

    assert platform.is_rocm()


def test_rocm_indexing_fallback(monkeypatch):
    index = importlib.import_module("minisgl.kernel.index")
    monkeypatch.setattr(index, "is_rocm", lambda: True)
    monkeypatch.setattr(
        index, "_jit_index_module", lambda *args, **kwargs: pytest.fail("loaded CUDA JIT")
    )

    weights = torch.arange(24, dtype=torch.float32).view(6, 4)
    indices = torch.tensor([2, 5, 1], dtype=torch.int32)
    output = torch.empty(3, 4)

    result = index.indexing(weights, indices, output=output)

    assert result is output
    torch.testing.assert_close(result, weights[indices.long()])


def test_rocm_masked_indexing_fallback(monkeypatch):
    index = importlib.import_module("minisgl.kernel.index")
    monkeypatch.setattr(index, "is_rocm", lambda: True)
    monkeypatch.setattr(
        index, "_jit_index_module", lambda *args, **kwargs: pytest.fail("loaded CUDA JIT")
    )

    weights = torch.arange(12, dtype=torch.float32).view(3, 4)
    indices = torch.tensor([3, 5, 6, 2], dtype=torch.int32)

    result = index.indexing(weights, indices, vocab_range=(3, 3))

    expected = torch.stack((weights[0], weights[2], torch.zeros(4), torch.zeros(4)))
    torch.testing.assert_close(result, expected)


def test_rocm_store_cache_fallback(monkeypatch):
    store = importlib.import_module("minisgl.kernel.store")
    monkeypatch.setattr(store, "is_rocm", lambda: True)
    monkeypatch.setattr(
        store, "_jit_store_module", lambda *args, **kwargs: pytest.fail("loaded CUDA JIT")
    )

    cache = torch.zeros(5, 2, 4)
    indices = torch.tensor([3, 1], dtype=torch.int32)
    k = torch.arange(8, dtype=torch.float32).view(2, 4)
    v = k + 10

    store.store_cache(cache[:, 0], cache[:, 1], indices, k, v)

    torch.testing.assert_close(cache[indices.long(), 0], k)
    torch.testing.assert_close(cache[indices.long(), 1], v)


def test_rocm_radix_fallback(monkeypatch):
    radix = importlib.import_module("minisgl.kernel.radix")
    monkeypatch.setattr(radix, "is_rocm", lambda: True)
    monkeypatch.setattr(radix, "_load_radix_module", lambda: pytest.fail("loaded TVM-FFI"))

    assert radix.fast_compare_key(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4])) == 2
    assert radix.fast_compare_key(torch.tensor([1, 2]), torch.tensor([1, 2, 3])) == 2


def test_rocm_nvtx_annotation_is_a_noop(monkeypatch):
    torch_utils = importlib.import_module("minisgl.utils.torch_utils")
    monkeypatch.setattr(torch_utils, "is_rocm", lambda: True)

    class Layer:
        @torch_utils.nvtx_annotate("Layer")
        def forward(self, value):
            return value + 1

    assert Layer().forward(2) == 3
