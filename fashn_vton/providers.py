"""Provider abstractions for pose detection and human parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from fashn_human_parser import FashnHumanParser

from .dwpose import DWposeDetector, draw_pose


class PoseDetectorProvider(ABC):
    """Interface for pose detectors used by the pipeline."""

    @abstractmethod
    def detect(self, rgb_image: np.ndarray) -> dict:
        """Run pose detection and return a pose dictionary."""

    @abstractmethod
    def render_grayscale(self, pose: dict, height: int, width: int) -> np.ndarray:
        """Render pose dictionary as a single-channel grayscale image."""


class DWPoseProvider(PoseDetectorProvider):
    """Default internal DWPose provider."""

    def __init__(self, checkpoints_dir: str, device: str):
        self.detector = DWposeDetector(checkpoints_dir=checkpoints_dir, device=device)

    def detect(self, rgb_image: np.ndarray) -> dict:
        # DWPose expects BGR.
        return self.detector(rgb_image[..., ::-1])

    def render_grayscale(self, pose: dict, height: int, width: int) -> np.ndarray:
        return draw_pose(pose, height, width, grayscale=True)


class ParserProvider(ABC):
    """Interface for parser backends used by the pipeline."""

    @abstractmethod
    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """Return a label-id segmentation map."""


class FashnHumanParserProvider(ParserProvider):
    """Default parser backend wrapping `fashn-human-parser`."""

    def __init__(self, parser: FashnHumanParser):
        self.parser = parser

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        return self.parser.predict(rgb_image)
