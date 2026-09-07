import itertools

import pytest
import torch
from minisgl.recommendation import BeamState, Catalog, expand
from minisgl.recommendation.search import remap_decode_window


def test_search_matches_exhaustive_reference_and_never_returns_illegal_sid():
    paths = [(2, 4, 7), (2, 5, 8), (3, 4, 8), (3, 5, 7)]
    catalog = Catalog([{"item_id": str(i), "token_ids": list(p)} for i, p in enumerate(paths)])
    trie, width = catalog.to("cpu"), 8
    generator = torch.Generator().manual_seed(7)
    logits = {
        prefix: torch.randn(len(catalog.token_ids), generator=generator)
        for d in range(3)
        for prefix in {p[:d] for p in paths}
    }
    state = BeamState(
        torch.zeros(1, 1),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, 3, dtype=torch.long),
    )
    for depth in range(3):
        inputs = torch.stack(
            [logits.get(tuple(p[:depth]), torch.zeros(6)) for p in state.tokens[0].tolist()]
        )[None]
        state = expand(inputs, state, torch.tensor([depth]), trie, width).state
    found = {
        tuple(path): score
        for path, score in zip(state.tokens[0].tolist(), state.scores[0].tolist())
        if torch.isfinite(torch.tensor(score))
    }
    expected = {}
    for path in paths:
        expected[path] = sum(
            logits[path[:d]].log_softmax(0)[catalog.token_ids.index(token)].item()
            for d, token in enumerate(path)
        )
    assert found == pytest.approx(expected)


def test_mixed_depth_batch_and_request_isolation():
    catalog = Catalog(
        [{"item_id": str(p), "token_ids": list(p)} for p in itertools.product([1, 2], repeat=2)]
    )
    state = BeamState(
        torch.tensor([[0.0, -torch.inf], [0.0, -torch.inf]]),
        torch.tensor([[0, 0], [1, 1]]),
        torch.tensor([[[0, 0], [0, 0]], [[1, 0], [1, 0]]]),
    )
    result = expand(
        torch.tensor([[[3.0, 0.0], [0.0, 0.0]], [[0.0, 3.0], [0.0, 0.0]]]),
        state,
        torch.tensor([0, 1]),
        catalog.to("cpu"),
        2,
    )
    assert result.state.tokens[0, 0, 0] == 1
    assert result.state.tokens[1, 0].tolist() == [1, 2]


def test_noise_changes_selection_but_not_scores():
    catalog = Catalog([{"item_id": str(i), "token_ids": [i]} for i in range(2)])
    state = BeamState(
        torch.zeros(1, 1),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, 1, dtype=torch.long),
    )
    logits = torch.tensor([[[3.0, 0.0]]])
    result = expand(
        logits, state, torch.tensor([0]), catalog.to("cpu"), 1, torch.tensor([[[0.0, 100.0]]])
    )
    assert result.next_tokens.item() == 1
    assert result.state.scores.item() == pytest.approx(logits.log_softmax(-1)[0, 0, 1].item())


def test_fork_and_permutation_only_remap_decode_indices():
    table = torch.arange(40).reshape(4, 10)
    old = table.clone()
    rows = torch.tensor([[0, 1], [2, 3]])
    remap_decode_window(table, rows, torch.tensor([[1, 1], [1, 0]]), torch.tensor([3, 5]), 3)
    assert torch.equal(table[0, :3], old[0, :3])
    assert torch.equal(table[:2, 3:5], old[[1, 1], 3:5])
    assert torch.equal(table[2:, 5:7], old[[3, 2], 5:7])


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"item_id": "a", "token_ids": [-1]}],
        [{"item_id": "a", "token_ids": [1]}, {"item_id": "b", "token_ids": [1]}],
        [{"item_id": "a", "token_ids": [1]}, {"item_id": "b", "token_ids": [1, 2]}],
    ],
)
def test_catalog_rejects_ambiguous_input(entries):
    with pytest.raises(ValueError):
        Catalog(entries)
