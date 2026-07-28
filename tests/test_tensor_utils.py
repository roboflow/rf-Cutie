"""Tests for tensor shape and probability helpers."""

from math import prod

import pytest
import torch

from cutie.utils.tensor_utils import aggregate, cls_to_one_hot, pad_divide_by, unpad


@pytest.mark.parametrize('shape', [(2, 3, 5), (1, 2, 3, 5), (1, 2, 3, 3, 5)])
def test_pad_divide_by_round_trips_supported_tensor_ranks(shape: tuple[int, ...]) -> None:
    """Padding then unpadding restores odd-sized tensors for each supported rank."""
    image = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)

    padded, padding = pad_divide_by(image, 4)

    assert padded.shape[-2:] == (4, 8)
    assert torch.equal(unpad(padded, padding), image)


def test_aggregate_adds_background_probability_as_logits() -> None:
    """Aggregation prepends the product background probability before logit conversion."""
    probabilities = torch.tensor([[[[0.2]], [[0.3]]]])

    logits = aggregate(probabilities, dim=1)

    expected = torch.logit(torch.tensor([[[[0.56]], [[0.2]], [[0.3]]]]))
    torch.testing.assert_close(logits, expected)


def test_cls_to_one_hot_preserves_each_class_position() -> None:
    """One-hot conversion creates exactly one active channel per input label."""
    class_ids = torch.tensor([[[[0, 2], [1, 2]]]])

    one_hot = cls_to_one_hot(class_ids, num_objects=2)

    assert one_hot.shape == (1, 3, 2, 2)
    assert torch.equal(one_hot.argmax(dim=1, keepdim=True), class_ids)
    assert torch.equal(one_hot.sum(dim=1), torch.ones_like(class_ids[:, 0]))
