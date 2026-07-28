"""Tests for deterministic memory similarity, affinity, and readout math."""

import torch

from cutie.model.utils.memory_utils import do_softmax, get_affinity, get_similarity, readout


def test_cpu_rng_starts_from_the_suite_seed() -> None:
    """The autouse fixture resets Torch randomness to the declared test seed."""
    values = torch.rand(4)

    torch.manual_seed(7)

    assert torch.equal(values, torch.rand(4))


def test_similarity_and_affinity_are_finite_and_normalized() -> None:
    """Similarity variants produce normalized per-query affinity weights."""
    memory_key = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    memory_shrinkage = torch.ones((1, 1, 2))
    query_key = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    query_selection = torch.ones((1, 2, 2))

    selected = get_similarity(memory_key, memory_shrinkage, query_key, query_selection)
    unselected = get_similarity(memory_key, memory_shrinkage, query_key, None)
    affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)

    assert selected.shape == unselected.shape == (1, 2, 2)
    assert torch.isfinite(selected).all()
    assert torch.isfinite(unselected).all()
    torch.testing.assert_close(affinity.sum(dim=1), torch.ones((1, 2)))


def test_top_k_softmax_and_readout_keep_only_selected_memory_entries() -> None:
    """Top-k affinity is sparse and readout follows its selected memory values."""
    similarity = torch.tensor([[[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]]])

    affinity, usage = do_softmax(similarity, top_k=1, return_usage=True)

    assert torch.equal(affinity, torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]))
    assert torch.equal(usage, torch.tensor([[0.0, 1.0, 1.0]]))
    memory_values = torch.tensor([[[[[10.0, 20.0]]]]])
    assert torch.equal(readout(affinity[:, :2], memory_values), torch.tensor([[[[20.0, 0.0]]]]))
