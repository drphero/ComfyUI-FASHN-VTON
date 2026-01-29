"""Sampling utilities for Rectified Flow inference."""

import math

import torch


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """
    Apply time shift to timesteps for flow matching schedule.

    Args:
        mu: Time shift parameter (controls schedule steepness)
        sigma: Sigma parameter (typically 1.0)
        t: Timestep tensor with values in (0, 1]

    Returns:
        Shifted timesteps
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_rf_schedule(num_steps: int, mu: float = 1.5, reverse: bool = True) -> list[float]:
    """
    Generate timestep schedule for Rectified Flow sampling.

    Creates a shifted linear schedule that provides better sample quality
    by spending more time at higher noise levels.

    Args:
        num_steps: Number of sampling steps
        mu: Time shift parameter (higher = more time at high noise)
        reverse: If True, returns schedule from t=0 to t=1 (for denoising)

    Returns:
        List of timesteps of length num_steps + 1
    """
    if reverse:
        mu = -mu
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = time_shift(mu, 1.0, timesteps)
    timesteps = timesteps.tolist()
    return timesteps[::-1] if reverse else timesteps
