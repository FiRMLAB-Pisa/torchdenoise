"""Give a denoiser the shapes and dtypes MRI actually produces.

DeepInverse denoisers take ``(batch, channel, *spatial)`` real tensors. MRI
gives complex volumes with any number of axes in front of the spatial ones --
slices, contrasts, echoes, dynamic frames, repetitions -- and a spatial
denoiser has nothing to say about those axes, so it must treat each of them
independently rather than mistake one for a spatial dimension.

Everything before the last ``spatial_dims`` axes is therefore folded into the
batch, which makes slice and time the same case and needs no argument to say
which is which. A 3D denoiser on ``(frames, slices, y, x)`` is wrong for the
same reason, and is refused rather than silently transformed: DeepInverse's
wavelet denoiser accepts a 2D tensor with ``wvdim=3`` and returns something.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["ComplexMode", "spatial_apply"]

ComplexMode = Literal["real_imag", "abs_angle"]


def _complex_wrapper(model: torch.nn.Module, mode: ComplexMode) -> torch.nn.Module:
    """Lift a real-valued denoiser to complex input, DeepInverse's way."""
    from deepinv.models import ComplexDenoiserWrapper

    return ComplexDenoiserWrapper(model, mode=mode)


def spatial_apply(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: float | torch.Tensor,
    *,
    spatial_dims: int,
    complex_mode: ComplexMode | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Apply a spatial denoiser to a stack of any shape.

    Parameters
    ----------
    model
        The denoiser, taking ``(batch, channel, *spatial)``.
    x
        Input, shaped ``(*independent, *spatial)``. Every leading axis is
        denoised independently, whether it is a slice, a contrast or a frame.
    sigma
        Noise level, passed through.
    spatial_dims
        How many trailing axes are spatial: 2 or 3.
    complex_mode
        How to handle complex input that ``model`` cannot take itself. ``None``
        means it can, and the tensor is passed through as it is.
    **kwargs
        Passed to the model.

    Returns
    -------
    torch.Tensor
        The denoised stack, in the shape and dtype it arrived in.

    Raises
    ------
    ValueError
        If the input has fewer axes than the denoiser has spatial dimensions,
        which would otherwise be silently reinterpreted.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise._adapt import spatial_apply
    >>> identity = lambda value, level, **_: value
    >>> stack = torch.zeros(3, 5, 16, 16)          # frames, slices, y, x
    >>> spatial_apply(identity, stack, 0.1, spatial_dims=2).shape
    torch.Size([3, 5, 16, 16])
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
    if x.ndim < spatial_dims:
        raise ValueError(
            f"a {spatial_dims}D denoiser needs at least {spatial_dims} axes, "
            f"got shape {tuple(x.shape)}"
        )

    spatial = x.shape[x.ndim - spatial_dims :]
    leading = x.shape[: x.ndim - spatial_dims]
    folded = x.reshape(-1, 1, *spatial)

    applied = model
    if x.is_complex() and complex_mode is not None:
        applied = _complex_wrapper(model, complex_mode)

    denoised = applied(folded, sigma, **kwargs)
    return denoised.reshape(*leading, *spatial).to(x.dtype)
