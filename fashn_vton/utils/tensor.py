"""PyTorch tensor utilities for image processing."""

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def numpy_to_torch(img: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image to torch tensor.

    For 3D arrays (H, W, C), permutes to (C, H, W).
    For 2D arrays (H, W), passes through unchanged.

    Args:
        img: Input numpy array of shape (H, W, C) or (H, W)

    Returns:
        Torch tensor of shape (C, H, W) or (H, W)
    """
    t = torch.from_numpy(img)
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    return t


def normalize_uint8_to_neg1_1(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize uint8 image tensor from [0, 255] to [-1, 1] range.

    Args:
        x: Input tensor with values in [0, 255]

    Returns:
        Normalized tensor with values in [-1, 1]
    """
    return x / 127.5 - 1.0


def _neg1_1_to_0_1(normed_img: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] normalized tensor to [0, 1] range."""
    return (normed_img + 1) * 0.5


def tensor_to_pil(img: torch.Tensor, unnormalize: bool = False) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image.

    Args:
        img: Input tensor of shape (C, H, W)
        unnormalize: If True, convert from [-1, 1] to [0, 1] range first

    Returns:
        PIL Image
    """
    if unnormalize:
        img = _neg1_1_to_0_1(img)
    return to_pil_image(img)


def unpack_images(x: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Unpack image patches back to full images.

    Used after transformer processing to convert patch representations
    back to spatial images.

    Args:
        x: Tensor of shape (batch_size, channels * patch_size^2, h, w)
        patch_size: Size of patches used during packing

    Returns:
        Tensor of shape (batch_size, channels, h * patch_size, w * patch_size)
    """
    return rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size)
