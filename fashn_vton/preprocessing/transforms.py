"""Image transforms for preprocessing."""

from typing import Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps


def _default(val, default_val):
    """Return val if not None, else default_val (or call it if callable)."""
    if val is not None:
        return val
    return default_val() if callable(default_val) else default_val


class AspectPreserveResize:
    """
    Resize images while preserving aspect ratio.

    Args:
        target_size: Target (width, height)
        mode: Resize mode
            - "fit": Scale to fit within target (may be smaller)
            - "exceed": Scale to exceed target (may be larger)
            - "short": Scale based on shorter dimension
            - "long": Scale based on longer dimension
        backend: "pil" or "opencv"
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        mode: Literal["short", "long", "fit", "exceed"] = "fit",
        backend: Literal["pil", "opencv"] = "pil",
    ):
        self.target_size = target_size
        self.mode = mode
        self.backend = backend

    def _get_or_infer_scale_factor(
        self, width: int, height: int, allow_upsampling: bool = True
    ) -> float:
        target_width, target_height = self.target_size
        scale_factor_width = target_width / width
        scale_factor_height = target_height / height

        if self.mode == "long":
            scale_factor = min(scale_factor_width, scale_factor_height)
        elif self.mode == "short":
            scale_factor = max(scale_factor_width, scale_factor_height)
        elif self.mode == "fit":
            scale_factor = min(scale_factor_width, scale_factor_height)
        elif self.mode == "exceed":
            scale_factor = max(scale_factor_width, scale_factor_height)
        else:
            raise ValueError("Invalid mode. It should be 'short', 'long', 'fit', or 'exceed'.")

        if not allow_upsampling and scale_factor > 1.0:
            return 1.0

        return scale_factor

    def _resize_image_pil(
        self, img: Image.Image, scale_factor: float, interpolation: Optional[int] = None
    ) -> Image.Image:
        if scale_factor == 1.0:
            return img

        width, height = img.size
        new_width = int(scale_factor * width)
        new_height = int(scale_factor * height)

        interpolation = _default(interpolation, Image.LANCZOS)
        return img.resize((new_width, new_height), interpolation)

    def _resize_image_opencv(
        self, img: np.ndarray, scale_factor: float, interpolation: Optional[int] = None
    ) -> np.ndarray:
        if scale_factor == 1.0:
            return img

        height, width = img.shape[:2]
        new_width = int(scale_factor * width)
        new_height = int(scale_factor * height)

        interpolation = _default(
            interpolation, cv2.INTER_LANCZOS4 if scale_factor > 1 else cv2.INTER_AREA
        )

        return cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray],
        allow_upsampling: bool = True,
        interpolation: Optional[int] = None,
    ) -> Union[Image.Image, np.ndarray]:
        if self.backend == "pil":
            width, height = img.size
        elif self.backend == "opencv":
            height, width = img.shape[:2]

        scale_factor = self._get_or_infer_scale_factor(width, height, allow_upsampling)

        if self.backend == "pil":
            return self._resize_image_pil(img, scale_factor, interpolation=interpolation)
        elif self.backend == "opencv":
            return self._resize_image_opencv(img, scale_factor, interpolation=interpolation)


class PadToShape:
    """
    Pad images to a target shape with symmetric padding.

    Args:
        target_size: Target (width, height)
        fill_value: Padding color (int for grayscale, tuple for RGB)
        backend: "pil" or "opencv"
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        fill_value: Union[int, tuple] = 0,
        backend: Literal["pil", "opencv"] = "opencv",
    ) -> None:
        self.target_width, self.target_height = target_size
        self.backend = backend
        if isinstance(fill_value, int):
            self.fill_value = (fill_value,) * 3
        else:
            self.fill_value = fill_value
        self.padding_mem: Optional[Tuple[int, int, int, int]] = None

    @staticmethod
    def _calculate_needed_padding(
        width: int, height: int, target_width: int, target_height: int
    ) -> Tuple[int, int, int, int]:
        total_width_padding = max(target_width - width, 0)
        total_height_padding = max(target_height - height, 0)

        pad_left = total_width_padding // 2
        pad_top = total_height_padding // 2
        pad_right = total_width_padding - pad_left
        pad_bottom = total_height_padding - pad_top

        return pad_left, pad_top, pad_right, pad_bottom

    def _pad_image_pil(self, img: Image.Image, padding: Tuple[int, int, int, int]) -> Image.Image:
        return ImageOps.expand(img, border=padding, fill=self.fill_value)

    def _pad_image_opencv(self, img: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
        pad_left, pad_top, pad_right, pad_bottom = padding
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=self.fill_value
        )

    def unpad(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """Remove padding using stored padding dimensions."""
        if self.padding_mem is None:
            raise ValueError("Padding memory is not set.")

        pad_left, pad_top, pad_right, pad_bottom = self.padding_mem

        if isinstance(img, Image.Image):
            return img.crop((pad_left, pad_top, img.width - pad_right, img.height - pad_bottom))
        return img[pad_top : img.shape[0] - pad_bottom, pad_left : img.shape[1] - pad_right]

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray],
        mem_padding: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        if self.backend == "pil":
            width, height = img.size
        else:
            height, width = img.shape[:2]

        padding = self._calculate_needed_padding(width, height, self.target_width, self.target_height)

        if mem_padding:
            self.padding_mem = padding

        if self.backend == "pil":
            return self._pad_image_pil(img, padding)
        return self._pad_image_opencv(img, padding)


class ResizePad:
    """
    Aspect-preserving resize followed by symmetric padding.

    Combines AspectPreserveResize and PadToShape to resize images to fit
    within target dimensions while preserving aspect ratio, then pads
    to reach exact target size.

    Args:
        target_image_size: Target (width, height)
        backend: "pil" or "opencv"
    """

    def __init__(
        self,
        target_image_size: Tuple[int, int],
        backend: Literal["pil", "opencv"] = "opencv",
    ) -> None:
        self.resize_fn = AspectPreserveResize(target_size=target_image_size, mode="fit", backend=backend)
        self.pad_fn = PadToShape(target_image_size, backend=backend)

    def unpad(self, img: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """Remove padding to restore original dimensions."""
        return self.pad_fn.unpad(img)

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray],
        mem_padding: bool = False,
        interpolation: Optional[int] = None,
    ) -> Union[Image.Image, np.ndarray]:
        img = self.resize_fn(img, interpolation=interpolation)
        return self.pad_fn(img, mem_padding=mem_padding)
