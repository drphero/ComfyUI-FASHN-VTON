"""Clothing-agnostic image creation."""

import logging
from typing import List, Optional

import numpy as np
from fashn_human_parser import BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS

from ..utils import setup_logger
from .masks import asymmetric_dilate_mask, create_bounded_mask, create_contour_following_mask, dilate_mask

# Re-export constants from fashn_human_parser for convenience
FASHN_LABELS_TO_IDS = LABELS_TO_IDS
BODY_COVERAGE_TO_FASHN_LABELS = BODY_COVERAGE_TO_LABELS
IDENTITY_FASHN_LABELS = tuple(IDENTITY_LABELS)


def _default(val, default_val):
    """Return val if not None, else default_val (or call it if callable)."""
    if val is not None:
        return val
    return default_val() if callable(default_val) else default_val


def _create_hybrid_contour_bounded_mask(
    contour_mask: np.ndarray,
    bounded_mask: np.ndarray,
    min_distance_threshold: float = 100.0,
    logger: Optional[logging.Logger] = None,
    baseline_height: float = 864.0,
) -> np.ndarray:
    """
    Create hybrid mask by removing over-aggressive bounded expansions.

    Combines contour-following and bounding-box masks, removing pixels from
    the bounded mask that are too far from the contour mask.

    Args:
        contour_mask: Precise contour-following mask
        bounded_mask: More aggressive bounded box mask
        min_distance_threshold: Max distance from contour for bounded pixels (at baseline height)
        logger: Optional logger instance
        baseline_height: Reference height for scaling threshold

    Returns:
        Hybrid mask with over-aggressive bounded pixels removed
    """
    import cv2

    logger = _default(logger, lambda: setup_logger("hybrid_mask"))

    # Scale threshold based on image height
    height_scale = contour_mask.shape[0] / baseline_height
    scaled_threshold = min_distance_threshold * height_scale

    if scaled_threshold <= 0:
        logger.debug("scaled_threshold<=0, returning pure contour mask")
        return contour_mask

    hybrid_mask = bounded_mask.copy()

    # Find pixels in bounded but not in contour (potential over-expansion)
    bounded_extra = bounded_mask & ~contour_mask

    if not np.any(bounded_extra):
        logger.debug("No extra pixels in bounded mask, returning bounded mask")
        return bounded_mask

    # Compute distance from bounded extra pixels to nearest contour mask pixel
    distance_from_contour = cv2.distanceTransform(
        (~contour_mask).astype(np.uint8), cv2.DIST_L2, 5
    )

    # Remove pixels too far from contour
    bounded_extra_coords = np.where(bounded_extra)
    extra_distances = distance_from_contour[bounded_extra]
    remove_mask = extra_distances > scaled_threshold
    remove_coords = (bounded_extra_coords[0][remove_mask], bounded_extra_coords[1][remove_mask])

    hybrid_mask[remove_coords] = False

    return hybrid_mask


def create_garment_image(
    img_np: np.ndarray,
    seg_pred: np.ndarray,
    labels_to_segment_indices: List[int],
    mask_value: int = 127,
    disable_masking: bool = False,
) -> np.ndarray:
    """
    Create garment image with optional masking.

    Masks out regions not belonging to the specified garment labels.

    Args:
        img_np: Input image array (will be modified in-place)
        seg_pred: Segmentation prediction array
        labels_to_segment_indices: List of label indices to keep
        mask_value: Value to fill masked regions (default: 127 gray)
        disable_masking: If True, return image unchanged

    Returns:
        Processed garment image array
    """
    if not disable_masking:
        selected_labels_mask = np.isin(seg_pred, labels_to_segment_indices)
        img_np[~selected_labels_mask] = mask_value

    return img_np


def create_clothing_agnostic_image(
    img_np: np.ndarray,
    seg_pred: np.ndarray,
    labels_to_segment_indices: List[int],
    body_coverage: str,
    mask_value: int = 127,
    disable_masking: bool = False,
    min_distance_threshold: float = 100.0,
    baseline_height: float = 864.0,
    mask_limbs: bool = True,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Create clothing-agnostic image.

    Masks garments and body parts based on the target category.

    Args:
        img_np: Input image array (will be modified in-place)
        seg_pred: Segmentation prediction array
        labels_to_segment_indices: List of label indices to mask
        body_coverage: Coverage type ("full", "upper", or "lower")
        mask_value: Value to fill masked regions (default: 127 gray)
        disable_masking: If True, return image unchanged
        min_distance_threshold: Distance threshold for hybrid mask (at baseline height)
        baseline_height: Reference height for parameter scaling
        mask_limbs: If True, also mask arms/legs based on body_coverage
        logger: Optional logger instance

    Returns:
        Clothing-agnostic image array
    """
    logger = _default(logger, lambda: setup_logger("clothing_agnostic"))

    if disable_masking:
        return img_np

    # Scale parameters based on image height
    height_scale = seg_pred.shape[0] / baseline_height
    logger.debug(f"Height scale factor: {height_scale:.3f} (height: {seg_pred.shape[0]})")

    # Add body parts to mask based on body coverage
    labels_ids_dict = FASHN_LABELS_TO_IDS.copy()
    if mask_limbs:
        if body_coverage in ("full", "upper"):
            labels_to_segment_indices += [labels_ids_dict["arms"], labels_ids_dict["torso"]]
        if body_coverage in ("full", "lower"):
            labels_to_segment_indices += [labels_ids_dict["legs"]]

    # Create base mask
    mask = np.isin(seg_pred, labels_to_segment_indices)

    # Buffer mask to avoid leaks
    scaled_buffer_kernel = max(1, int(4 * height_scale))
    buffer_mask = dilate_mask(mask, kernel=(scaled_buffer_kernel, scaled_buffer_kernel))

    # Create bounded mask
    bounded_mask = create_bounded_mask(mask)

    # Create contour following mask
    scaled_brush_radius = max(1, int(18 * height_scale))
    contour_mask = create_contour_following_mask(mask, brush_radius=scaled_brush_radius)

    # Create hybrid mask
    ca_mask = _create_hybrid_contour_bounded_mask(
        contour_mask, bounded_mask, logger=logger, min_distance_threshold=min_distance_threshold
    )

    # Apply asymmetric dilation for inpainting workspace
    scaled_right = int(33 * height_scale)
    scaled_left = int(33 * height_scale)
    scaled_up = int(16 * height_scale)
    scaled_down = int(16 * height_scale)
    ca_mask = asymmetric_dilate_mask(ca_mask, right=scaled_right, left=scaled_left, up=scaled_up, down=scaled_down)

    # Create exclusion mask (regions to preserve)
    identity_ids = [labels_ids_dict[label] for label in IDENTITY_FASHN_LABELS]

    # Conditional identity based on coverage
    if body_coverage == "upper":
        identity_ids.append(labels_ids_dict["legs"])
    elif body_coverage == "lower":
        identity_ids.append(labels_ids_dict["arms"])

    exclusion_mask = np.isin(seg_pred, identity_ids)

    # Handle hands and feet
    if body_coverage in ("full", "upper"):
        hands_mask = seg_pred == labels_ids_dict["hands"]
        exclusion_mask = exclusion_mask | hands_mask

    if body_coverage in ("full", "lower"):
        feet_mask = seg_pred == labels_ids_dict["feet"]
        exclusion_mask = exclusion_mask | feet_mask

    final_mask = buffer_mask | (ca_mask & ~exclusion_mask)
    img_np[final_mask] = mask_value

    return img_np
