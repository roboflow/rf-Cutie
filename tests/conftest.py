"""Shared deterministic fixtures for the CPU unit suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_torch_random_seed() -> None:
    """Reset Torch's CPU RNG before every test."""
    torch.manual_seed(7)
