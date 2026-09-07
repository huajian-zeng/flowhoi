# This code is based on https://github.com/openai/guided-diffusion
# Copyright (c) 2021 OpenAI; see third_party/licenses/guided-diffusion-MIT.txt.
# Modified for FlowHOI's inference utilities.
"""Neural-network utilities for diffusion models."""

import math

import torch as th
import torch.nn as nn


class SiLU(nn.Module):
    """SiLU activation, x * sigmoid(x)."""
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """Group normalization computed in float32 with input-dtype output."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)

def sum_flat(tensor):
    """Sum over all non-batch dimensions."""
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """Create a 32-group normalization layer."""
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create [N, dim] sinusoidal embeddings for fractional timesteps."""
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """Trade backward recomputation for lower activation memory when enabled."""
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    """Autograd function for activation checkpointing."""
    @staticmethod
    @th.amp.custom_fwd(device_type='cuda')
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_length = length
        ctx.save_for_backward(*args)
        with th.no_grad():
            output_tensors = ctx.run_function(*args[:length])
        return output_tensors

    @staticmethod
    @th.amp.custom_fwd(device_type='cuda')
    def backward(ctx, *output_grads):
        args = list(ctx.saved_tensors)

        input_indices = [i for (i, x) in enumerate(args) if x.requires_grad]
        if not input_indices:
            return (None, None) + tuple(None for _ in args)

        with th.enable_grad():
            for i in input_indices:
                if i < ctx.input_length:
                    args[i] = args[i].detach().requires_grad_()
                    args[i] = args[i].view_as(args[i])
            output_tensors = ctx.run_function(*args[:ctx.input_length])

        if isinstance(output_tensors, th.Tensor):
            output_tensors = [output_tensors]

        out_and_grads = [(o, g) for (o, g) in zip(output_tensors, output_grads) if o.requires_grad]
        if not out_and_grads:
            return (None, None) + tuple(None for _ in args)

        computed_grads = th.autograd.grad(
            [o for (o, g) in out_and_grads],
            [args[i] for i in input_indices],
            [g for (o, g) in out_and_grads]
        )

        input_grads = [None for _ in args]
        for (i, g) in zip(input_indices, computed_grads):
            input_grads[i] = g
        return (None, None) + tuple(input_grads)
