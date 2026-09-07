from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DeviceTrie:
    offsets: torch.Tensor
    columns: torch.Tensor
    children: torch.Tensor
    token_ids: torch.Tensor
    edge_range: torch.Tensor

    def successors(self, nodes: torch.Tensor):
        starts, ends = self.offsets[nodes], self.offsets[nodes + 1]
        edges = starts[..., None] + self.edge_range
        valid = edges < ends[..., None]
        edges = edges.clamp_max(self.columns.numel() - 1)
        return self.columns[edges], self.children[edges], valid


class Catalog:
    """Fixed-depth, unique token paths. CSR storage is O(nodes + edges), not O(NV)."""

    def __init__(self, entries: list[dict]):
        if not isinstance(entries, list) or not entries:
            raise ValueError("Catalog must not be empty")
        self.items: dict[tuple[int, ...], str] = {}
        item_ids = set()
        for entry in entries:
            if not isinstance(entry, dict) or "item_id" not in entry or "token_ids" not in entry:
                raise ValueError("Each catalog entry needs item_id and token_ids")
            tokens = entry["token_ids"]
            item_id = str(entry["item_id"])
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)
            ):
                raise ValueError("token_ids must be a nonempty list of nonnegative integers")
            key = tuple(tokens)
            if key in self.items or item_id in item_ids:
                raise ValueError("Catalog requires unique item IDs and unique SID token paths")
            self.items[key] = item_id
            item_ids.add(item_id)
        depths = {len(x) for x in self.items}
        if len(depths) != 1:
            raise ValueError(
                "All SIDs must have the same token depth (include boundaries explicitly)"
            )
        self.depth = depths.pop()
        self.token_ids = sorted({t for path in self.items for t in path})
        column = {token: i for i, token in enumerate(self.token_ids)}
        nodes: list[dict[int, int]] = [{}]
        for path in sorted(self.items):
            node = 0
            for token in path:
                if token not in nodes[node]:
                    nodes[node][token] = len(nodes)
                    nodes.append({})
                node = nodes[node][token]
        self.offsets, self.columns, self.children = [0], [], []
        for node in nodes:
            for token, child in sorted(node.items()):
                self.columns.append(column[token])
                self.children.append(child)
            self.offsets.append(len(self.columns))
        self.max_degree = max(map(len, nodes))

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text()))

    def to(self, device: torch.device | str) -> DeviceTrie:
        def tensor(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        return DeviceTrie(
            tensor(self.offsets),
            tensor(self.columns),
            tensor(self.children),
            tensor(self.token_ids),
            torch.arange(self.max_degree, device=device),
        )
