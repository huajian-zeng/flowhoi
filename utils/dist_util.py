# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Device selection helpers."""

import torch as th

used_device = 0


def setup_dist(device=0):
    """Record which CUDA device the run should use."""
    global used_device
    used_device = device


def dev():
    """The device to run on."""
    if th.cuda.is_available() and used_device >= 0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """Load a PyTorch checkpoint."""
    return th.load(path, **kwargs)
