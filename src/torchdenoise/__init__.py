"""Denoisers for MRI: DeepInverse's classical ones, and a locally low-rank one.

The classical denoisers are DeepInverse's, wrapped only so that MRI data goes
straight in. What the wrapper adds is the two things MRI always needs and a
denoiser written for natural images never has: complex data, and any number of
axes in front of the spatial ones. Slices, contrasts, echoes and dynamic frames
are all the same case to a spatial denoiser -- axes it has nothing to say about
and must therefore treat independently.

The locally low-rank denoiser is not DeepInverse's. It thresholds the singular
values of small blocks taken across the non-spatial axis, so it is the one that
uses the correlation between contrasts or frames rather than working around it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._adapt import ComplexMode, spatial_apply
from ._classical import TGV, TV, Bilateral, Median, Wavelet, WaveletDict

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "TGV",
    "TV",
    "Bilateral",
    "ComplexMode",
    "Median",
    "Wavelet",
    "WaveletDict",
    "__version__",
    "spatial_apply",
]
