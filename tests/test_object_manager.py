"""Tests for stable object IDs and compact temporary IDs."""

import torch

from cutie.inference.object_manager import ObjectManager


def test_object_lifecycle_compacts_temporary_ids_and_remaps_masks() -> None:
    """Adding, deleting, and purging objects maintains stable external IDs."""
    manager = ObjectManager()

    assert manager.add_new_objects([10, 30, 50]) == ([1, 2, 3], [10, 30, 50])
    assert manager.add_new_objects(10) == ([1], [10])

    manager.delete_objects(30)

    assert manager.get_tmp_to_obj_mapping() == {10: 1, 50: 2}
    assert torch.equal(manager.tmp_to_obj_cls(torch.tensor([[0, 1, 2]])), torch.tensor([[0, 10, 50]]))

    manager.find_object_by_id(50).poke()
    manager.find_object_by_id(50).poke()
    purged, temporary_ids, object_ids = manager.purge_inactive_objects(1)

    assert purged
    assert temporary_ids == [1]
    assert object_ids == [10]
    assert manager.all_obj_ids == [10]
    assert manager.make_one_hot(torch.tensor([[0, 10, 50]])).tolist() == [[[False, True, False]]]
