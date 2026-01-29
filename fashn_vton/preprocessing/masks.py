"""Mask processing utilities."""

import cv2
import numpy as np


def dilate_mask(mask: np.ndarray, kernel: tuple = (33, 33), iterations: int = 1) -> np.ndarray:
    """
    Dilate the mask to create a buffer zone around the selected areas.

    Args:
        mask: Input binary mask
        kernel: Dilation kernel size
        iterations: Number of dilation iterations

    Returns:
        Dilated boolean mask
    """
    kernel = np.ones(kernel, np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    return dilated_mask.astype(bool)


def create_bounded_mask(mask: np.ndarray) -> np.ndarray:
    """
    Create a mask that fills the bounding box of the input mask.

    Args:
        mask: Input binary mask

    Returns:
        Bounded mask filling the bounding rectangle
    """
    bounded_mask = np.zeros_like(mask)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    bounded_mask[y : y + h, x : x + w] = 1
    return bounded_mask


def asymmetric_dilate_mask(
    mask: np.ndarray, right: int, left: int, up: int, down: int
) -> np.ndarray:
    """
    Dilate mask asymmetrically in different directions.

    Args:
        mask: Input binary mask
        right: Dilation amount to the right
        left: Dilation amount to the left
        up: Dilation amount upward
        down: Dilation amount downward

    Returns:
        Asymmetrically dilated boolean mask
    """
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    kernel_width = left + right + 1
    kernel_height = up + down + 1
    kernel = np.ones((kernel_height, kernel_width), np.uint8)

    anchor_x = right
    anchor_y = down

    mask = cv2.dilate(mask, kernel, anchor=(anchor_x, anchor_y))

    return mask.astype(bool)


def create_contour_following_mask(
    mask: np.ndarray,
    brush_radius: int = 36,
    smoothing_sigma: float | None = None,
    supersample: int = 1,
    keep_holes: bool = False,
) -> np.ndarray:
    """
    Inflate mask so it looks like it was painted with a large soft brush.

    Uses signed distance field for smooth contour following with optional
    supersampling for ultra-clean edges.

    Args:
        mask: Input segmentation (foreground != 0)
        brush_radius: Extra pixels the virtual brush extends beyond the garment
        smoothing_sigma: Edge-smoothing sigma (Gaussian blur). If None, defaults to brush_radius / 2.5
        supersample: 1 = fastest; 2-4 for ultra-clean edges via upscale + max-pool downsample
        keep_holes: If True, preserve interior holes (e.g. neck opening)

    Returns:
        Boolean mask, same H×W as input, guaranteed to contain the original mask
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    if smoothing_sigma is None:
        smoothing_sigma = brush_radius / 2.5

    if supersample < 1 or not isinstance(supersample, int):
        raise ValueError("`supersample` must be a positive integer.")

    # Optional super-sampling
    if supersample > 1:
        mask_work = cv2.resize(
            mask.astype(np.uint8),
            dsize=None,
            fx=supersample,
            fy=supersample,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        br = brush_radius * supersample
        sig = smoothing_sigma * supersample
    else:
        mask_work = mask.copy()
        br = brush_radius
        sig = smoothing_sigma

    # Dilation ensures superset
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * br + 1, 2 * br + 1))
    dilated = cv2.dilate(mask_work.astype(np.uint8), se).astype(bool)

    # Signed distance field
    dist_out = cv2.distanceTransform((~dilated).astype(np.uint8), cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(dilated.astype(np.uint8), cv2.DIST_L2, 5)
    signed = dist_out - dist_in  # <0 inside, >0 outside

    # Smooth the level set
    signed_blur = cv2.GaussianBlur(signed, (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
    smooth = signed_blur <= 0  # level-set 0

    # Optional hole fill
    if not keep_holes:
        smooth = _fill_holes_cv(smooth)

    # Containment: ensure original mask is included
    smooth |= mask_work

    # Downsample if supersampled
    if supersample > 1:
        smooth = _max_pool_downsample(smooth, supersample)

    return smooth.astype(bool)


def _max_pool_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-wise max-pooling downsample (factor must divide both axes)."""
    h, w = arr.shape
    if h % factor or w % factor:
        raise ValueError("Supersample factor must divide mask dimensions.")
    arr = arr.reshape(h // factor, factor, w // factor, factor)
    return arr.any(axis=(1, 3))


def _fill_holes_cv(binary: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask using flood-fill."""
    h, w = binary.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    inv = (~binary).astype(np.uint8)  # holes & background = 1
    flood = inv.copy()
    cv2.floodFill(flood, flood_mask, (0, 0), 0)  # erase the true background
    holes = flood == 1
    return binary | holes
