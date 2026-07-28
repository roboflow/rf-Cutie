"""Tests for key/value memory lifetime and object cleanup."""

import torch

from cutie.inference.kv_memory_store import KeyValueMemoryStore


def _memory_tensors(
    offset: float,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Create two-entry tensors with distinct values for memory state assertions."""
    key = torch.tensor([[[offset, offset + 1], [offset + 2, offset + 3]]])
    values = {1: key + 10, 2: key + 20}
    shrinkage = torch.ones((1, 1, 2))
    selection = key + 30
    return key, values, shrinkage, selection


def test_memory_store_tracks_permanent_temporary_usage_and_objects() -> None:
    """Memory keeps permanent entries while clearing temporary state and purging objects."""
    store = KeyValueMemoryStore(save_selection=True, save_usage=True)
    first_key, first_values, first_shrinkage, first_selection = _memory_tensors(0)
    store.add(first_key, first_values, first_shrinkage, first_selection, as_permanent='first')

    second_key, second_values, second_shrinkage, second_selection = _memory_tensors(100)
    store.add(second_key, second_values, second_shrinkage, second_selection)

    assert store.perm_size(0) == 2
    assert store.non_perm_size(0) == 2
    store.update_bucket_usage(0, torch.tensor([[0.0, 0.0, 1.0, 2.0]]))
    key, _, selection, values, usage = store.get_all_sliced(0, 0, 0)
    assert key.shape[-1] == 2
    assert selection.shape[-1] == 2
    assert sorted(values) == [1, 2]
    assert torch.all(usage > 0)

    store.clear_non_permanent_memory()

    assert store.size(0) == 2
    assert store.selection[0].shape[-1] == 0
    store.purge_except([1])
    assert store.num_objects == 1
    assert 2 not in store
    assert store.buckets[0] == [1]
