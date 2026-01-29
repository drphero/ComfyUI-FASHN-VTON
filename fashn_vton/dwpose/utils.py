"""
Drawing utilities adapted from DWPose (Apache-2.0):
https://github.com/IDEA-Research/DWPose
"""

import math

import cv2
import matplotlib
import numpy as np

eps = 0.01


def draw_bodypose_gray(canvas, candidate, subset):
    H, W = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18],
    ]

    base_values = np.linspace(20, 240, 18).astype(np.uint8)
    stickwidth = 4
    limb_canvas = np.zeros_like(canvas)

    for i in range(17):
        limb_gray = int(base_values[i])
        for n in range(len(subset)):
            indexA = int(subset[n][limbSeq[i][0] - 1])
            indexB = int(subset[n][limbSeq[i][1] - 1])
            if indexA == -1 or indexB == -1:
                continue

            xA = int(candidate[indexA][0] * W)
            yA = int(candidate[indexA][1] * H)
            xB = int(candidate[indexB][0] * W)
            yB = int(candidate[indexB][1] * H)

            mX = (xA + xB) // 2
            mY = (yA + yB) // 2
            length = int(math.hypot(xA - xB, yA - yB))
            angle = int(math.degrees(math.atan2(yA - yB, xA - xB)))

            polygon = cv2.ellipse2Poly((mX, mY), (length // 2, stickwidth), angle, 0, 360, 1)
            cv2.fillConvexPoly(limb_canvas, polygon, limb_gray)

    canvas = np.maximum(canvas, (limb_canvas * 0.6).astype(np.uint8))

    keypoint_to_limb_map = {}
    for i, (a, b) in enumerate(limbSeq):
        if a not in keypoint_to_limb_map:
            keypoint_to_limb_map[a] = []
        if b not in keypoint_to_limb_map:
            keypoint_to_limb_map[b] = []
        keypoint_to_limb_map[a].append(i)
        keypoint_to_limb_map[b].append(i)

    for i in range(1, 19):
        if i not in keypoint_to_limb_map:
            continue

        connected_limbs = keypoint_to_limb_map[i]
        if connected_limbs:
            point_gray = min(255, int(base_values[connected_limbs[0]] * 1.3))
        else:
            point_gray = 200

        for n in range(len(subset)):
            index = int(subset[n][i - 1])
            if index == -1:
                continue

            x = int(candidate[index][0] * W)
            y = int(candidate[index][1] * H)
            cv2.circle(canvas, (x, y), 4, point_gray, thickness=-1)

    return canvas


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18], [3, 17], [6, 18],
    ]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
        [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=2,
                )

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_handpose_gray(canvas, all_hand_peaks):
    H, W = canvas.shape

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
        [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    edge_values = np.linspace(160, 220, len(edges)).astype(np.uint8)

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), int(edge_values[ie]), thickness=2)

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, 240, thickness=-1)
    return canvas


def draw_facepose_gray(canvas, all_lmks):
    H, W = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, 200, thickness=-1)
    return canvas
