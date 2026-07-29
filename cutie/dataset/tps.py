import numpy as np
from PIL import Image
import cv2

from thin_plate_spline import ThinPlateSpline

cv2.setNumThreads(0)


def pick_random_points(h, w, n_samples):
    """Select independent normalized x and y coordinates without replacement."""
    y_idx = np.random.choice(np.arange(h), size=n_samples, replace=False)
    x_idx = np.random.choice(np.arange(w), size=n_samples, replace=False)
    return y_idx / h, x_idx / w


def inverse_tps_grid(c_src, c_dst, dshape):
    """Return normalized source coordinates for every destination image pixel."""
    height, width = dshape[:2]
    destination_x, destination_y = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height))
    destination_grid = np.stack((destination_x, destination_y), axis=-1)

    spline = ThinPlateSpline()
    spline.fit(c_dst, c_src)
    return spline.transform(destination_grid.reshape(-1, 2)).reshape(height, width, 2)


def warp_dual_cv(img, mask, c_src, c_dst):
    """Warp an image and mask with a shared TPS coordinate map."""
    grid = inverse_tps_grid(c_src, c_dst, img.shape)
    mapx = (grid[:, :, 0] * img.shape[1]).astype(np.float32)
    mapy = (grid[:, :, 1] * img.shape[0]).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR), cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)


def random_tps_warp(img, mask, scale, n_ctrl_pts=12):
    """Apply a random NumPy-driven TPS warp to an image and its mask."""
    img = np.asarray(img)
    mask = np.asarray(mask)

    h, w = mask.shape
    points = pick_random_points(h, w, n_ctrl_pts)
    c_src = np.stack(points, 1)
    c_dst = c_src + np.random.normal(scale=scale, size=c_src.shape)
    warp_im, warp_gt = warp_dual_cv(img, mask, c_src, c_dst)

    return Image.fromarray(warp_im), Image.fromarray(warp_gt)
