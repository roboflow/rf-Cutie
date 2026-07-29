"""Tests for thin-plate-spline image augmentation."""

import numpy as np
import pytest

pytest.importorskip('tps', reason='thin-plate-spline is supplied by the train dependency profile.')

from cutie.dataset.tps import inverse_tps_grid, warp_dual_cv


def test_inverse_tps_grid_maps_destination_control_points_to_source_points() -> None:
    """The dense inverse map interpolates every fixed control-point correspondence."""
    destination = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    source = destination + np.array([0.05, -0.03])

    grid = inverse_tps_grid(source, destination, (3, 3, 3))

    np.testing.assert_allclose(grid.reshape(-1, 2), source, atol=1e-8)


def test_warp_dual_cv_preserves_image_shape_and_discrete_mask_labels() -> None:
    """Image and mask warps share a map while retaining nearest-neighbor mask labels."""
    image = np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3)
    mask = np.arange(81, dtype=np.uint8).reshape(9, 9) % 3
    destination = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=np.float64,
    )
    source = destination + np.array([0.02, -0.01])

    warped_image, warped_mask = warp_dual_cv(image, mask, source, destination)

    assert warped_image.shape == image.shape
    assert warped_image.dtype == image.dtype
    assert warped_mask.shape == mask.shape
    assert set(np.unique(warped_mask)).issubset({0, 1, 2})
