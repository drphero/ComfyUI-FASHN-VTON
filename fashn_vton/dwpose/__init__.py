"""
DWPose - Effective Whole-body Pose Estimation

This module is adapted from the IDEA-Research/DWPose repository (onnx branch):
https://github.com/IDEA-Research/DWPose/tree/onnx/ControlNet-v1-1-nightly/annotator/dwpose

Original paper:
    "Effective Whole-body Pose Estimation with Two-stages Distillation"
    Zhendong Yang, Ailing Zeng, Chun Yuan, Yu Li
    ICCV 2023, CV4Metaverse Workshop
    https://arxiv.org/abs/2307.15880

License: Apache-2.0
"""

from .dwpose import DWposeDetector, draw_pose
