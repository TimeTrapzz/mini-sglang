"""Normalize scalar/per-channel FP8 checkpoints before projection merging.

Runtime W8A8 uses its own dynamic activation scales. Stored input_scale values
are therefore intentionally discarded, not used as weight scales.
"""

from contextlib import ExitStack

import safetensors
import torch


def checkpoint_tensors(files, device):
    with ExitStack() as stack:
        readers = [
            stack.enter_context(safetensors.safe_open(f, framework="pt", device=str(device)))
            for f in files
        ]
        index = {name: reader for reader in readers for name in reader.keys()}
        for name, reader in index.items():
            if name.startswith(("vision_tower.", "multi_modal_projector.")):
                continue
            if name.endswith((".weight_scale", ".input_scale")):
                continue
            if name.endswith((".weight_scale_inv", ".input_scale_inv")):
                raise ValueError(
                    "Inverse/block FP8 scales are unsupported; use BF16 or weight_scale"
                )
            weight = reader.get_tensor(name)
            if weight.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
            ):
                key = name.removesuffix(".weight") + ".weight_scale"
                if not name.endswith(".weight") or key not in index:
                    raise ValueError(f"FP8 tensor {name} needs an explicit weight_scale")
                scale = index[key].get_tensor(key).float()
                if weight.ndim != 2 or scale.numel() not in (1, weight.shape[0]):
                    raise ValueError(f"{key} must contain one scale or one per output channel")
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"{key} must be finite and positive")
                weight = (weight.float() * scale.reshape(-1, 1)).to(torch.bfloat16)
            yield name, weight
