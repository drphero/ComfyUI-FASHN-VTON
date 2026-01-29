"""Model checkpoint loading utilities."""

import os

import torch
from safetensors.torch import load_file


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load model checkpoint from local file or HuggingFace Hub.

    Supports:
    - Local .pt, .pth files (PyTorch format)
    - Local .safetensors files
    - HuggingFace repo IDs (e.g., "fashn-ai/fashn-vton-1.5")

    Args:
        checkpoint_path: Local file path or HuggingFace repo ID
        device: Device to load the checkpoint to

    Returns:
        The loaded state dictionary
    """
    # Check if it's a local file
    if os.path.isfile(checkpoint_path):
        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        elif checkpoint_path.endswith(".safetensors"):
            return load_file(checkpoint_path, device=device)
        else:
            raise ValueError(f"Unknown checkpoint file format: {checkpoint_path}")

    # Check if it looks like a HuggingFace repo ID
    if "/" in checkpoint_path and not checkpoint_path.endswith((".pt", ".pth", ".safetensors")):
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="model.safetensors",
        )
        return load_file(local_path, device=device)

    raise ValueError(f"Checkpoint not found: {checkpoint_path}")
