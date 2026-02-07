"""Pose keypoint format conversion helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

KEYPOINT_VISIBILITY_THRESHOLD = 0.3
BODY25_TO_COCO18 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def _coerce_pose_payload(payload: Any) -> Any:
    """Convert common object payloads into plain Python structures."""
    if payload is None:
        return None

    if isinstance(payload, (dict, list, tuple, np.ndarray)):
        return payload

    for method_name in ("to_dict", "dict", "model_dump"):
        method = getattr(payload, method_name, None)
        if callable(method):
            try:
                data = method()
                if data is not None:
                    return data
            except Exception:
                pass

    if hasattr(payload, "__dict__"):
        return vars(payload)

    return payload


def _extract_canvas_size(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Find width/height metadata if present."""
    width_keys = ("canvas_width", "image_width", "width", "w")
    height_keys = ("canvas_height", "image_height", "height", "h")

    width = None
    height = None

    for key in width_keys:
        if key in data:
            width = float(data[key])
            break
    for key in height_keys:
        if key in data:
            height = float(data[key])
            break

    return width, height


def _to_triplets(values: Any) -> Optional[np.ndarray]:
    """Convert flat keypoint vectors into (N,3)."""
    if values is None:
        return None

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return None
        arr = arr.reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] >= 3:
        arr = arr[:, :3]
    else:
        return None

    return arr


def _normalize_xy_inplace(points: np.ndarray, width: Optional[float], height: Optional[float]) -> None:
    """Normalize x/y to [0, 1] when input appears to be in pixel coordinates."""
    if points.size == 0:
        return

    x_max = float(np.nanmax(points[:, 0]))
    y_max = float(np.nanmax(points[:, 1]))
    appears_normalized = x_max <= 1.5 and y_max <= 1.5
    if appears_normalized:
        return

    scale_w = width if width and width > 1 else max(x_max, 1.0)
    scale_h = height if height and height > 1 else max(y_max, 1.0)

    points[:, 0] = points[:, 0] / float(scale_w)
    points[:, 1] = points[:, 1] / float(scale_h)


def _compute_person_area(points: np.ndarray, valid: np.ndarray) -> float:
    valid_points = points[valid]
    if valid_points.size == 0:
        return 0.0

    valid_x = valid_points[:, 0][valid_points[:, 0] > 0]
    valid_y = valid_points[:, 1][valid_points[:, 1] > 0]
    if valid_x.size == 0 or valid_y.size == 0:
        return 0.0
    return float((valid_x.max() - valid_x.min()) * (valid_y.max() - valid_y.min()))


def _select_best_person(candidate: np.ndarray, scores: np.ndarray) -> int:
    """Mimic internal DWPose single-person preference (largest confident body)."""
    valid_keypoints = scores[:, 1:14] > KEYPOINT_VISIBILITY_THRESHOLD
    headless_scores = np.sum(scores[:, 1:14] * valid_keypoints, axis=1)

    areas: List[float] = []
    for idx in range(candidate.shape[0]):
        area = _compute_person_area(candidate[idx, 1:14], valid_keypoints[idx])
        areas.append(area)

    score_area = headless_scores * np.asarray(areas, dtype=np.float32)
    score_area[~np.isfinite(score_area)] = 0

    if np.all(score_area == 0):
        return int(np.argmax(headless_scores))
    return int(np.argmax(score_area))


def _build_dwpose_dict(
    body_candidates: np.ndarray,
    body_scores: np.ndarray,
    hands: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Create canonical DWPose-style dictionary from body/hands/faces arrays."""
    num_people = body_candidates.shape[0]
    body = body_candidates[:, :18, :2].astype(np.float32)
    score = body_scores[:, :18].astype(np.float32)

    subset = np.full((num_people, 18), -1, dtype=np.float32)
    for i in range(num_people):
        for j in range(18):
            visible = score[i, j] > KEYPOINT_VISIBILITY_THRESHOLD
            if visible:
                subset[i, j] = int(18 * i + j)
            else:
                body[i, j] = -1

    pose: Dict[str, Any] = {"bodies": {"candidate": body.reshape(num_people * 18, 2), "subset": subset}}

    if hands is not None and hands.size:
        pose["hands"] = hands.astype(np.float32)
    if faces is not None and faces.size:
        pose["faces"] = faces.astype(np.float32)

    return pose


def _convert_people_keypoints(data: Dict[str, Any], single_person: bool = True) -> Optional[Dict[str, Any]]:
    """Convert OpenPose-like payload (`people`) into DWPose-style format."""
    people = data.get("people")
    if not isinstance(people, list) or not people:
        return None

    width, height = _extract_canvas_size(data)

    body_candidates = []
    body_scores = []
    all_hands: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    per_person_hand_ranges: List[Tuple[int, int]] = []
    per_person_face_ranges: List[Tuple[int, int]] = []

    for person in people:
        if not isinstance(person, dict):
            continue

        body_triplets = _to_triplets(
            person.get("pose_keypoints_2d") or person.get("body_keypoints_2d")
        )
        if body_triplets is None:
            continue

        if body_triplets.shape[0] >= 25:
            body_triplets = body_triplets[BODY25_TO_COCO18]
        elif body_triplets.shape[0] < 18:
            continue
        else:
            body_triplets = body_triplets[:18]

        _normalize_xy_inplace(body_triplets, width, height)
        body_triplets[:, :2] = np.clip(body_triplets[:, :2], -1.0, 2.0)

        body_candidates.append(body_triplets[:, :2].copy())
        body_scores.append(body_triplets[:, 2].copy())

        hand_start = len(all_hands)
        for hand_key in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
            hand_triplets = _to_triplets(person.get(hand_key))
            if hand_triplets is None or hand_triplets.shape[0] < 21:
                continue
            hand_triplets = hand_triplets[:21]
            _normalize_xy_inplace(hand_triplets, width, height)
            hand = hand_triplets[:, :2].copy()
            hand[hand_triplets[:, 2] <= KEYPOINT_VISIBILITY_THRESHOLD] = -1
            all_hands.append(hand)
        hand_end = len(all_hands)
        per_person_hand_ranges.append((hand_start, hand_end))

        face_start = len(all_faces)
        face_triplets = _to_triplets(person.get("face_keypoints_2d"))
        if face_triplets is not None and face_triplets.shape[0] > 0:
            _normalize_xy_inplace(face_triplets, width, height)
            face = face_triplets[:, :2].copy()
            face[face_triplets[:, 2] <= KEYPOINT_VISIBILITY_THRESHOLD] = -1
            all_faces.append(face)
        face_end = len(all_faces)
        per_person_face_ranges.append((face_start, face_end))

    if not body_candidates:
        return None

    candidate_np = np.stack(body_candidates, axis=0)
    score_np = np.stack(body_scores, axis=0)

    if single_person and candidate_np.shape[0] > 1:
        best_idx = _select_best_person(candidate_np, score_np)
        candidate_np = candidate_np[best_idx : best_idx + 1]
        score_np = score_np[best_idx : best_idx + 1]

        hand_start, hand_end = per_person_hand_ranges[best_idx]
        face_start, face_end = per_person_face_ranges[best_idx]
        hands_np = (
            np.stack(all_hands[hand_start:hand_end], axis=0).astype(np.float32)
            if hand_end > hand_start
            else None
        )
        faces_np = (
            np.stack(all_faces[face_start:face_end], axis=0).astype(np.float32)
            if face_end > face_start
            else None
        )
        return _build_dwpose_dict(candidate_np, score_np, hands=hands_np, faces=faces_np)

    hands_np = np.stack(all_hands, axis=0).astype(np.float32) if all_hands else None
    faces_np = np.stack(all_faces, axis=0).astype(np.float32) if all_faces else None
    return _build_dwpose_dict(candidate_np, score_np, hands=hands_np, faces=faces_np)


def _convert_dwpose_like(data: Dict[str, Any], single_person: bool = True) -> Optional[Dict[str, Any]]:
    """Normalize already DWPose-like payloads into canonical dict."""
    bodies = data.get("bodies")
    if not isinstance(bodies, dict):
        return None

    candidate = np.asarray(bodies.get("candidate"), dtype=np.float32)
    subset = np.asarray(bodies.get("subset"), dtype=np.float32)

    if candidate.ndim == 2 and candidate.shape[1] >= 2 and candidate.shape[0] % 18 == 0:
        num_people = max(1, candidate.shape[0] // 18)
        candidate_people = candidate[:, :2].reshape(num_people, 18, 2)
    elif candidate.ndim == 3 and candidate.shape[1] >= 18 and candidate.shape[2] >= 2:
        candidate_people = candidate[:, :18, :2]
        num_people = candidate_people.shape[0]
    elif candidate.ndim == 2 and candidate.shape[0] == 18 and candidate.shape[1] >= 2:
        candidate_people = candidate[:, :2][None, ...]
        num_people = 1
    else:
        return None

    if subset.ndim == 1 and subset.shape[0] >= 18:
        subset = subset[:18][None, ...]
    elif subset.ndim == 2 and subset.shape[1] >= 18:
        subset = subset[:, :18]
    else:
        subset = np.full((num_people, 18), -1, dtype=np.float32)

    if subset.shape[0] != num_people:
        if subset.shape[0] == 1 and num_people > 1:
            subset = np.repeat(subset, num_people, axis=0)
        else:
            subset = subset[:num_people]
            if subset.shape[0] < num_people:
                pad = np.full((num_people - subset.shape[0], 18), -1, dtype=np.float32)
                subset = np.concatenate([subset, pad], axis=0)

    width, height = _extract_canvas_size(data)
    for i in range(num_people):
        _normalize_xy_inplace(candidate_people[i], width, height)

    subset_is_score_like = np.all((subset == -1) | ((subset >= 0) & (subset <= 1.0 + 1e-6)))
    if subset_is_score_like:
        score_np = np.where(subset >= 0, subset, 0).astype(np.float32)
    else:
        score_np = (subset >= 0).astype(np.float32)

    if single_person and num_people > 1:
        best_idx = _select_best_person(candidate_people, score_np)
        candidate_people = candidate_people[best_idx : best_idx + 1]
        score_np = score_np[best_idx : best_idx + 1]

    hands = data.get("hands")
    faces = data.get("faces")
    hands_np = np.asarray(hands, dtype=np.float32) if hands is not None else None
    faces_np = np.asarray(faces, dtype=np.float32) if faces is not None else None

    return _build_dwpose_dict(candidate_people, score_np, hands=hands_np, faces=faces_np)


def convert_pose_keypoints_to_dwpose(payload: Any, single_person: bool = True) -> Optional[Dict[str, Any]]:
    """
    Convert arbitrary pose-keypoint payloads to canonical internal DWPose dictionary.

    Supported forms:
    - DWPose-like dict: `{\"bodies\": {\"candidate\": ..., \"subset\": ...}, ...}`
    - OpenPose-like dict with `people` payload
    - Objects exposing `to_dict()`/`dict()`/`model_dump()`
    """
    data = _coerce_pose_payload(payload)
    if data is None:
        return None

    if isinstance(data, tuple) and len(data) == 1:
        data = data[0]
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        return None

    converted = _convert_dwpose_like(data, single_person=single_person)
    if converted is not None:
        return converted

    return _convert_people_keypoints(data, single_person=single_person)
