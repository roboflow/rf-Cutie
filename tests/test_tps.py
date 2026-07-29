"""Tests for thin-plate-spline numerical behavior and integration."""

import numpy as np
from PIL import Image
import pytest

from thin_plate_spline import ThinPlateSpline
from cutie.dataset import tps as tps_module


def _expected_identity_normalized_grid(dshape) -> np.ndarray:
    height, width = dshape[:2]
    destination_x, destination_y = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height))
    return np.stack((destination_x, destination_y), axis=-1)


def test_tps_backend_identity_grid_matches_hard_values() -> None:
    """Identity control-point pairs should reproduce the normalized sampling grid."""
    my_pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    destination_grid = _expected_identity_normalized_grid((3, 4, 3))
    tps = ThinPlateSpline()
    tps.fit(my_pts, my_pts)
    out = tps.transform(destination_grid.reshape(-1, 2)).reshape((3, 4, 2))

    np.testing.assert_allclose(out, destination_grid, atol=1e-8)


def test_tps_backend_map_respects_control_points_and_will_match_cutie_wrapper() -> None:
    """Warp map should keep control-point correspondences and match cutie wrapper output."""
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
    height, width = 3, 3
    destination_x, destination_y = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height))
    destination_grid = np.stack((destination_x, destination_y), axis=-1).reshape(-1, 2)

    spline = ThinPlateSpline()
    spline.fit(destination, source)
    reference = spline.transform(destination_grid).reshape(height, width, 2)
    wrapped = tps_module.inverse_tps_grid(source, destination, (height, width, 3))

    np.testing.assert_allclose(reference.reshape(-1, 2), wrapped.reshape(-1, 2), atol=1e-8)


def test_tps_affine_case_produces_expected_center_mapping() -> None:
    """A known affine-style control configuration maps the center to a hard-coded point."""
    destination = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    source = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.1, 1.0],
            [1.1, 1.0],
        ],
        dtype=np.float64,
    )
    expected_center = np.array([0.55, 0.5], dtype=np.float64)

    tps = ThinPlateSpline()
    tps.fit(destination, source)
    direct_out = tps.transform(np.array([[0.5, 0.5]], dtype=np.float64))
    wrapped_grid = tps_module.inverse_tps_grid(source, destination, (3, 3, 3))
    wrapped_center = wrapped_grid[1, 1]

    np.testing.assert_allclose(direct_out.reshape(-1, 2), expected_center, atol=1e-8)
    np.testing.assert_allclose(wrapped_center, expected_center, atol=1e-8)


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

    grid = tps_module.inverse_tps_grid(source, destination, (3, 3, 3))

    np.testing.assert_allclose(grid.reshape(-1, 2), destination + np.array([0.05, -0.03]), atol=1e-8)


def test_warp_dual_cv_preserves_image_shape_and_discrete_mask_labels() -> None:
    """Image and mask warps share a map while retaining nearest-neighbor mask labels."""
    image = np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3)
    mask = np.arange(81, dtype=np.uint8).reshape(9, 9) % 3
    destination = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=np.float64,
    )
    source = destination + np.array([0.02, -0.01])

    warped_image, warped_mask = tps_module.warp_dual_cv(image, mask, source, destination)

    assert warped_image.shape == image.shape
    assert warped_image.dtype == image.dtype
    assert warped_mask.shape == mask.shape
    assert set(np.unique(warped_mask)).issubset({0, 1, 2})


def test_warp_dual_cv_with_identity_control_points_is_stable() -> None:
    """Identity control-point pairs keep pixel grid order and mask labels unchanged."""
    image = np.array([[[0, 10, 20], [30, 40, 50]], [[60, 70, 80], [90, 100, 110]]], dtype=np.uint8)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    control_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    warped_image, warped_mask = tps_module.warp_dual_cv(image, mask, control_points, control_points)

    np.testing.assert_array_equal(warped_image, image)
    np.testing.assert_array_equal(warped_mask, mask)


def test_pick_random_points_respects_bounds_and_uniqueness() -> None:
    """Picked coordinates are normalized and sampled without replacement per axis."""
    np.random.seed(0)
    h, w, n_samples = 7, 11, 5

    ys, xs = tps_module.pick_random_points(h, w, n_samples)

    assert ys.shape == (n_samples,)
    assert xs.shape == (n_samples,)
    assert np.all(ys >= 0) and np.all(ys < 1)
    assert np.all(xs >= 0) and np.all(xs < 1)
    assert len(np.unique((ys * h).astype(int))) == n_samples
    assert len(np.unique((xs * w).astype(int))) == n_samples


def test_random_tps_warp_outputs_pil_images_with_matching_shapes() -> None:
    """Random TPS warp returns PIL images aligned to the original resolution."""
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:6, :6] = [10, 20, 30]
    image[6:, 6:] = [40, 50, 60]
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:6] = 1
    mask[:, 6:] = 2

    np.random.seed(0)
    warped_image, warped_mask = tps_module.random_tps_warp(image, mask, scale=0.02, n_ctrl_pts=3)

    assert isinstance(warped_image, Image.Image)
    assert isinstance(warped_mask, Image.Image)
    assert np.array(warped_image).shape == image.shape
    assert np.array(warped_mask).shape == mask.shape
    assert np.array(warped_image).dtype == np.uint8
    assert np.array(warped_mask).dtype == np.uint8
    assert set(np.unique(np.array(warped_mask))).issubset({0, 1, 2})


def test_random_tps_warp_rejects_too_many_control_points_for_source_grid() -> None:
    """The helper should fail for invalid control-point counts via NumPy's sampler."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match='cannot take a larger sample'):
        tps_module.random_tps_warp(image, mask, scale=0.02, n_ctrl_pts=5)


def test_random_tps_warp_is_reproducible_for_fixed_seed() -> None:
    """Equal seeds must produce identical TPS outputs for both image and mask."""
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = (np.arange(16, dtype=np.uint8).reshape(4, 4) % 2).astype(np.uint8)

    np.random.seed(0)
    image_a, mask_a = tps_module.random_tps_warp(image, mask, scale=0.02, n_ctrl_pts=3)

    np.random.seed(0)
    image_b, mask_b = tps_module.random_tps_warp(image, mask, scale=0.02, n_ctrl_pts=3)

    np.testing.assert_array_equal(np.array(image_a), np.array(image_b))
    np.testing.assert_array_equal(np.array(mask_a), np.array(mask_b))
