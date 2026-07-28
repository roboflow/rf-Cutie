"""Tests for point-based feature sampling and uncertainty helpers."""

import torch

from cutie.utils.point_features import (
    calculate_uncertainty,
    cat,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


def test_cat_preserves_a_single_tensor_and_concatenates_multiple_tensors() -> None:
    """Single-tensor input remains an identity while multi-tensor input is concatenated."""
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[3.0, 4.0]])

    assert cat([first]) is first
    assert torch.equal(cat([first, second], dim=0), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_calculate_uncertainty_handles_class_agnostic_binary_and_multiclass_logits() -> None:
    """Uncertainty is the negative prediction margin for each supported class layout."""
    class_agnostic_logits = torch.tensor([[[[4.0, -2.0]]]])
    binary_logits = torch.tensor([[[[4.0, -2.0]], [[-3.0, 0.5]]]])
    multiclass_logits = torch.tensor([[[[2.0, -1.0]], [[1.0, 3.0]], [[0.0, 2.0]]]])

    class_agnostic_uncertainty = calculate_uncertainty(class_agnostic_logits)
    binary_uncertainty = calculate_uncertainty(binary_logits)
    multiclass_uncertainty = calculate_uncertainty(multiclass_logits)

    assert torch.equal(class_agnostic_uncertainty, torch.tensor([[[[-4.0, -2.0]]]]))
    assert torch.equal(binary_uncertainty, torch.tensor([[[[-3.0, -0.5]]]]))
    assert torch.equal(multiclass_uncertainty, torch.tensor([[[[-1.0, -1.0]]]]))


def test_point_sample_reads_normalized_corner_and_center_coordinates() -> None:
    """Normalized coordinates sample the corresponding feature-map corners and center."""
    feature_map = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    coordinates = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]])

    sampled = point_sample(feature_map, coordinates, align_corners=True)

    assert sampled.shape == (1, 1, 3)
    assert torch.equal(sampled, torch.tensor([[[1.0, 4.0, 2.5]]]))


def test_uncertain_point_sampling_is_seeded_and_stays_in_normalized_space() -> None:
    """Mixed importance and random sampling produces deterministic normalized point batches."""
    logits = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[0.5, -0.5], [-0.5, 0.5]]]])

    torch.manual_seed(11)
    first = get_uncertain_point_coords_with_randomness(
        logits,
        calculate_uncertainty,
        num_points=5,
        oversample_ratio=3,
        importance_sample_ratio=0.6,
    )
    torch.manual_seed(11)
    second = get_uncertain_point_coords_with_randomness(
        logits,
        calculate_uncertainty,
        num_points=5,
        oversample_ratio=3,
        importance_sample_ratio=0.6,
    )

    assert first.shape == (2, 5, 2)
    assert torch.equal(first, second)
    assert bool(torch.all((first >= 0.0) & (first <= 1.0)))
