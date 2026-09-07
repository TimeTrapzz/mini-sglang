from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .catalog import DeviceTrie


class BeamState(NamedTuple):
    scores: torch.Tensor  # [requests, beams], unperturbed cumulative log probabilities
    nodes: torch.Tensor
    tokens: torch.Tensor  # [requests, beams, SID depth]


class Expansion(NamedTuple):
    state: BeamState
    parents: torch.Tensor
    next_tokens: torch.Tensor


def expand(
    logits: torch.Tensor,
    state: BeamState,
    steps: torch.Tensor,
    trie: DeviceTrie,
    width: int,
    noise: torch.Tensor | None = None,
    fused: bool = False,
) -> Expansion:
    """Fixed-shape tensor-only expansion, safe to capture in a CUDA/HIP graph.

    Scores use log-softmax over the catalog's token vocabulary *before* trie masking.
    Noise only changes this step's selection, never the reported sequence scores.
    No EOS heuristic: terminal depth comes from the validated fixed-depth catalog.
    """
    batch = logits.shape[0]
    columns, children, valid = trie.successors(state.nodes)
    if fused and logits.is_cuda and logits.shape[-1] <= 32768:
        from .triton_ops import legal_scores

        scores = legal_scores(logits, state, trie)
    else:
        logprobs = F.log_softmax(logits.float(), dim=-1)
        scores = logprobs.gather(-1, columns) + state.scores[..., None]
        scores = scores.masked_fill(~valid, -torch.inf)
    flat = scores.flatten(1)
    rank = flat if noise is None else flat + noise.flatten(1)
    # The root may have fewer children than the beam width. Dead beams stay -inf.
    padding = max(0, width - flat.shape[1])
    selected = F.pad(rank, (0, padding), value=-torch.inf).topk(width, dim=1).indices
    real = selected < flat.shape[1]
    selected = selected.clamp_max(flat.shape[1] - 1)
    new_scores = flat.gather(1, selected).masked_fill(~real, -torch.inf)
    parents = selected // columns.shape[-1]
    new_nodes = children.flatten(1).gather(1, selected)
    tokens = trie.token_ids[columns.flatten(1).gather(1, selected)]
    history = state.tokens.gather(1, parents[..., None].expand(-1, -1, state.tokens.shape[-1]))
    history.scatter_(2, steps[:, None, None].expand(batch, width, 1), tokens[..., None])
    return Expansion(BeamState(new_scores, new_nodes, history), parents, tokens)


def remap_decode_window(
    table: torch.Tensor,
    rows: torch.Tensor,
    parents: torch.Tensor,
    prompt_lens: torch.Tensor,
    depth: int,
) -> None:
    """Share physical KV slots; copy only decode indices, never prefix KV tensors."""
    columns = prompt_lens[:, None, None] + torch.arange(depth - 1, device=table.device)
    sources = rows.gather(1, parents)
    # Advanced indexing materializes the RHS, so permutation/fork cannot overwrite its source.
    table[rows[..., None], columns] = table[sources[..., None], columns]
