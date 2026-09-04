"""The classical denoisers, wrapped so MRI data goes straight in.

Each is DeepInverse's, and the wrapper adds only what MRI needs: any number of
leading axes, denoised independently, and complex input. Which denoisers can
take complex themselves and which cannot is not documented anywhere, so it is
recorded here as measured -- the wavelet denoiser has an ``is_complex`` mode and
uses it, while total variation raises ``clamp is not supported for complex
types`` and is lifted by DeepInverse's own complex wrapper instead.
"""

from __future__ import annotations

from typing import Any

import torch

from ._adapt import ComplexMode, spatial_apply

__all__ = ["TGV", "TV", "Bilateral", "Median", "Wavelet", "WaveletDict"]


def _models() -> Any:
    from deepinv import models

    return models


class _Thresholded(torch.nn.Module):
    """Name the level ``sigma``, and forget the last call unless told not to.

    Two things about DeepInverse's variation denoisers only show up in use.
    They call the level ``ths`` and raise if it goes unset, while the complex
    wrapper passes it as ``sigma``. And they keep their iterates between calls,
    restarting only when the shape changes, so the same input denoises
    differently depending on what was denoised before it -- which inside an
    iterative reconstruction means the denoiser has a memory nobody asked for.
    """

    def __init__(self, inner: torch.nn.Module, *, warm_start: bool = False) -> None:
        super().__init__()
        self.inner = inner
        self.warm_start = warm_start

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Denoise, naming the level whichever way the inner denoiser wants."""
        if not self.warm_start:
            self.inner.restart = True
        return self.inner(x, ths=sigma, **kwargs)


class _Classical(torch.nn.Module):
    """A DeepInverse denoiser, plus the shapes and dtypes MRI brings."""

    spatial_dims: int
    complex_mode: ComplexMode | None

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        spatial_dims: int,
        complex_mode: ComplexMode | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.spatial_dims = spatial_dims
        self.complex_mode = complex_mode

    def _prepare(self, x: torch.Tensor) -> None:
        """Settle anything about the model that depends on the input."""

    def forward(
        self, x: torch.Tensor, sigma: float | torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Denoise a stack of any shape."""
        self._prepare(x)
        return spatial_apply(
            self.model,
            x,
            sigma,
            spatial_dims=self.spatial_dims,
            complex_mode=self.complex_mode,
            **kwargs,
        )


class Wavelet(_Classical):
    """Orthogonal-wavelet soft thresholding.

    Complex data uses DeepInverse's own complex transform, which thresholds a
    coefficient's magnitude and so keeps it whole, rather than shrinking its
    real and imaginary parts apart. Which of the two is used is decided at
    construction there -- it also fixes which axes are transformed -- so a
    model is built for each dtype as it is first seen rather than one being
    reconfigured. ``complex_mode`` forces the real/imaginary route instead;
    the two differ by around 20% of the signal amplitude at a threshold that
    removes a fifth of it.

    Parameters
    ----------
    spatial_dims
        2 or 3. Leading axes are denoised independently either way, so a 2D
        wavelet on a multi-slice multi-contrast stack is per slice and
        contrast, and a 3D one on the same stack is per contrast.
    wavelet, level, non_linearity
        As DeepInverse's ``WaveletDenoiser``.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import Wavelet
    >>> stack = torch.rand(2, 4, 32, 32, dtype=torch.complex64)
    >>> Wavelet()(stack, 0.05).shape
    torch.Size([2, 4, 32, 32])
    >>> Wavelet(spatial_dims=3)(torch.rand(2, 8, 32, 32), 0.05).shape
    torch.Size([2, 8, 32, 32])
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 2,
        wavelet: str = "db8",
        level: int = 3,
        non_linearity: str = "soft",
        complex_mode: ComplexMode | None = None,
        **kwargs: Any,
    ) -> None:
        self._settings = dict(
            wvdim=spatial_dims,
            wv=wavelet,
            level=level,
            non_linearity=non_linearity,
            **kwargs,
        )
        super().__init__(
            _models().WaveletDenoiser(is_complex=False, **self._settings),
            spatial_dims=spatial_dims,
            complex_mode=complex_mode,
        )
        self._real_model = self.model
        self._complex_model: torch.nn.Module | None = None
        self._native_complex = complex_mode is None

    def _prepare(self, x: torch.Tensor) -> None:
        """Use the transform built for this dtype.

        ``is_complex`` settles which axes DeepInverse transforms, so it cannot
        be changed on a built model: doing so leaves the transform pointed at a
        spatial axis and the real/imaginary pair.
        """
        if not self._native_complex:
            return
        if not x.is_complex():
            self.model = self._real_model
            return
        if self._complex_model is None:
            self._complex_model = _models().WaveletDenoiser(
                is_complex=True, **self._settings
            )
        self.model = self._complex_model


class WaveletDict(_Classical):
    """Several wavelets at once, averaged.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import WaveletDict
    >>> WaveletDict()(torch.rand(2, 16, 16), 0.05).shape
    torch.Size([2, 16, 16])
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 2,
        wavelets: tuple[str, ...] = ("db4", "db8"),
        level: int = 3,
        **kwargs: Any,
    ) -> None:
        model = _models().WaveletDictDenoiser(
            wvdim=spatial_dims, list_wv=list(wavelets), level=level, **kwargs
        )
        super().__init__(model, spatial_dims=spatial_dims, complex_mode="real_imag")


class TV(_Classical):
    """Total variation.

    DeepInverse's total variation cannot take complex input at all -- it clamps,
    and ``clamp`` has no meaning for a complex number -- so complex data is
    lifted by DeepInverse's own complex wrapper. ``real_imag``, the default,
    denoises the real and imaginary parts separately; ``abs_angle`` denoises
    magnitude and phase, which keeps a smooth phase smooth but wraps.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import TV
    >>> TV(iterations=5)(torch.rand(3, 16, 16, dtype=torch.complex64), 0.05).dtype
    torch.complex64
    """

    def __init__(
        self,
        *,
        complex_mode: ComplexMode = "real_imag",
        iterations: int = 100,
        warm_start: bool = False,
        **kwargs: Any,
    ) -> None:
        model = _Thresholded(
            _models().TVDenoiser(n_it_max=iterations, **kwargs),
            warm_start=warm_start,
        )
        super().__init__(model, spatial_dims=2, complex_mode=complex_mode)


class TGV(_Classical):
    """Total generalised variation, which does not mistake a ramp for an edge.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import TGV
    >>> TGV(iterations=5)(torch.rand(2, 16, 16), 0.05).shape
    torch.Size([2, 16, 16])
    """

    def __init__(
        self,
        *,
        complex_mode: ComplexMode = "real_imag",
        iterations: int = 100,
        warm_start: bool = False,
        **kwargs: Any,
    ) -> None:
        model = _Thresholded(
            _models().TGVDenoiser(n_it_max=iterations, **kwargs),
            warm_start=warm_start,
        )
        super().__init__(model, spatial_dims=2, complex_mode=complex_mode)


class Median(_Classical):
    """Median filter.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import Median
    >>> Median()(torch.rand(2, 16, 16), 0.05).shape
    torch.Size([2, 16, 16])
    """

    def __init__(
        self, *, complex_mode: ComplexMode = "real_imag", **kwargs: Any
    ) -> None:
        super().__init__(
            _models().MedianFilter(**kwargs), spatial_dims=2, complex_mode=complex_mode
        )


class Bilateral(_Classical):
    """Bilateral filter, which smooths within an edge but not across it.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import Bilateral
    >>> Bilateral()(torch.rand(2, 16, 16), 0.05).shape
    torch.Size([2, 16, 16])
    """

    def __init__(
        self, *, complex_mode: ComplexMode = "real_imag", **kwargs: Any
    ) -> None:
        super().__init__(
            _models().BilateralFilter(**kwargs),
            spatial_dims=2,
            complex_mode=complex_mode,
        )
