import numpy as np
import torch

from .utils import (
    draw_bodypose,
    draw_bodypose_gray,
    draw_facepose,
    draw_facepose_gray,
    draw_handpose,
    draw_handpose_gray,
)
from .wholebody import Wholebody

__all__ = ["DWposeDetector", "draw_pose"]

# Minimum confidence threshold for keypoint visibility
KEYPOINT_VISIBILITY_THRESHOLD = 0.3

def draw_pose(pose, H, W, canvas_value: int = 0, grayscale: bool = False):
    bodies = pose["bodies"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    if grayscale:
        draw_bodypose_fn = draw_bodypose_gray
        draw_handpose_fn = draw_handpose_gray
        draw_facepose_fn = draw_facepose_gray
        canvas = np.full((H, W), canvas_value, dtype=np.uint8)
    else:
        draw_bodypose_fn = draw_bodypose
        draw_handpose_fn = draw_handpose
        draw_facepose_fn = draw_facepose
        canvas_value = int(canvas_value / 0.6) if canvas_value > 0 else 0
        canvas = np.full((H, W, 3), canvas_value, dtype=np.uint8)

    canvas = draw_bodypose_fn(canvas, candidate, subset)

    if "hands" in pose:
        canvas = draw_handpose_fn(canvas, pose.get("hands"))
    if "faces" in pose:
        canvas = draw_facepose_fn(canvas, pose.get("faces"))

    return canvas


class DWposeDetector:
    def __init__(
        self,
        checkpoints_dir,
        device="cuda:0",
    ):

        self.pose_estimation = Wholebody(checkpoints_dir=checkpoints_dir, device=device)

    def _find_best_candidate(self, subset, candidate, score_threshold=KEYPOINT_VISIBILITY_THRESHOLD):
        # Apply score threshold to subset keypoints
        valid_keypoints = subset[:, 1:14] > score_threshold

        # Calculate scores for each candidate, only counting valid keypoints
        headless_scores = np.sum(subset[:, 1:14] * valid_keypoints, axis=1)

        # Extract keypoints for each candidate, excluding the head
        headless_keypoints = candidate[:, 1:14]

        def compute_area(keypoints):
            # Filter keypoints based on the valid_keypoints mask
            valid_kp = keypoints[
                valid_keypoints[0]
            ]  # Assuming all candidates have the same validity mask for simplicity
            valid_x, valid_y = valid_kp[:, 0][valid_kp[:, 0] > 0], valid_kp[:, 1][valid_kp[:, 1] > 0]
            if not len(valid_x) or not len(valid_y):
                return 0
            return (np.max(valid_x) - np.min(valid_x)) * (np.max(valid_y) - np.min(valid_y))

        areas = [compute_area(kp) for kp in headless_keypoints]

        # Here, we multiply scores by areas, but we need to handle division by zero or invalid calculations
        with np.errstate(divide="ignore", invalid="ignore"):
            scores_times_areas = headless_scores * np.array(areas)

        # Replace NaN or inf with 0 for np.nanargmax to work correctly
        scores_times_areas[np.isnan(scores_times_areas) | np.isinf(scores_times_areas)] = 0

        # If all scores are zero (or invalid), we might want to handle this case differently
        if np.all(scores_times_areas == 0):
            best_candidate_idx = np.argmax(headless_scores)
        else:
            best_candidate_idx = np.nanargmax(scores_times_areas)

        return (
            candidate[best_candidate_idx : best_candidate_idx + 1],
            subset[best_candidate_idx : best_candidate_idx + 1],
        )

    @torch.inference_mode()
    def __call__(self, oriImg: np.array, single: bool = True) -> dict:
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape

        candidate, subset = self.pose_estimation(oriImg)
        nums, keys, locs = candidate.shape

        if single and nums > 1:
            candidate, subset = self._find_best_candidate(subset, candidate)
            nums = 1  # Now we only have one candidate

        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        body = candidate[:, :18].copy()
        body = body.reshape(nums * 18, locs)
        score = subset[:, :18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > KEYPOINT_VISIBILITY_THRESHOLD:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1

        un_visible = subset < KEYPOINT_VISIBILITY_THRESHOLD
        candidate[un_visible] = -1

        foot = candidate[:, 18:24]
        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        return pose
