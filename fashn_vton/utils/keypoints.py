"""Keypoint utilities for pose detection."""

import numpy as np


def get_dummy_dw_keypoints() -> dict:
    """
    Get dummy DWPose keypoints dictionary for flat-lay garments.

    Returns a pose dictionary with all keypoints set to -1 to indicate
    no person is present (used for flat-lay garment images).

    Returns:
        Dictionary with 'bodies' key containing dummy keypoints
    """
    pose = {}
    pose["bodies"] = {"candidate": (-1) * np.ones((18, 2)), "subset": -1 * np.ones((1, 18))}
    return pose
