"""The classical denoisers, on the shapes and dtypes MRI produces."""

from __future__ import annotations

import pytest
import torch

from torchdenoise import TGV, TV, Bilateral, Median, Wavelet, WaveletDict

SPATIAL_2D = (32, 32)


def build(name: str, **kwargs):
    """Each denoiser, with enough iterations to be quick."""
    return {
        "wavelet": lambda: Wavelet(**kwargs),
        "waveletdict": lambda: WaveletDict(**kwargs),
        "tv": lambda: TV(iterations=5, **kwargs),
        "tgv": lambda: TGV(iterations=5, **kwargs),
        # the default 9x9 kernel erases a 32-pixel phantom
        "median": lambda: Median(kernel_size=3, **kwargs),
        "bilateral": lambda: Bilateral(**kwargs),
    }[name]()


ALL = ["wavelet", "waveletdict", "tv", "tgv", "median", "bilateral"]


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("shape", [SPATIAL_2D, (4, *SPATIAL_2D), (3, 4, *SPATIAL_2D)])
def test_shape_and_dtype_survive(name, dtype, shape) -> None:
    """Any number of leading axes, real or complex, in and out unchanged."""
    generator = torch.Generator().manual_seed(0)
    if dtype.is_complex:
        image = torch.rand(shape, generator=generator) + 1j * torch.rand(
            shape, generator=generator
        )
    else:
        image = torch.rand(shape, generator=generator)
    out = build(name)(image, 0.05)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


@pytest.mark.parametrize("name", ALL)
def test_leading_axes_are_independent(name) -> None:
    """A stack denoises exactly as its entries do one at a time."""
    generator = torch.Generator().manual_seed(1)
    stack = torch.rand(3, 2, *SPATIAL_2D, generator=generator)
    denoiser = build(name)
    together = denoiser(stack, 0.1)
    apart = torch.stack(
        [
            torch.stack([denoiser(stack[outer, inner], 0.1) for inner in range(2)])
            for outer in range(3)
        ]
    )
    assert torch.allclose(together, apart, atol=1e-5)


def test_three_dimensional_wavelet_uses_the_volume() -> None:
    """A 3D wavelet couples slices; a 2D one on the same stack does not."""
    generator = torch.Generator().manual_seed(2)
    volume = torch.rand(8, 32, 32, generator=generator)
    volumetric = Wavelet(spatial_dims=3)(volume, 0.1)
    per_slice = Wavelet(spatial_dims=2)(volume, 0.1)
    assert volumetric.shape == volume.shape
    assert not torch.allclose(volumetric, per_slice, atol=1e-3)


def test_three_dimensional_denoiser_refuses_a_plane() -> None:
    """Silently treating a 2D image as a volume is the trap this closes."""
    with pytest.raises(ValueError, match="needs at least 3 axes"):
        Wavelet(spatial_dims=3)(torch.rand(32, 32), 0.1)


def test_the_same_object_takes_both_dtypes() -> None:
    """is_complex settles which axes are transformed, so a model is built per dtype."""
    denoiser = Wavelet()
    assert denoiser(torch.rand(2, *SPATIAL_2D), 0.05).dtype == torch.float32
    complex_image = torch.rand(2, *SPATIAL_2D) + 1j * torch.rand(2, *SPATIAL_2D)
    assert denoiser(complex_image, 0.05).dtype == torch.complex64
    # and back again, which a reconfigured model would not survive
    assert denoiser(torch.rand(2, *SPATIAL_2D), 0.05).dtype == torch.float32


def test_complex_wavelet_routes_differ() -> None:
    """Thresholding a coefficient whole is not thresholding its parts apart."""
    generator = torch.Generator().manual_seed(3)
    image = torch.rand(2, *SPATIAL_2D, generator=generator) + 1j * torch.rand(
        2, *SPATIAL_2D, generator=generator
    )
    whole = Wavelet()(image, 0.2)
    apart = Wavelet(complex_mode="real_imag")(image, 0.2)
    assert (whole - apart).abs().max() > 0.05 * image.abs().max()


@pytest.mark.parametrize("name", ALL)
def test_denoising_reduces_noise(name) -> None:
    """The point of the exercise."""
    generator = torch.Generator().manual_seed(4)
    rows, columns = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    truth = ((rows**2 + columns**2) < 0.5).float().expand(2, 32, 32)
    noisy = truth + 0.1 * torch.randn(truth.shape, generator=generator)
    cleaned = build(name)(noisy, 0.1)
    assert (cleaned - truth).norm() < (noisy - truth).norm()


def test_spatial_dims_must_be_two_or_three() -> None:
    from torchdenoise import spatial_apply

    with pytest.raises(ValueError, match="must be 2 or 3"):
        spatial_apply(lambda v, s: v, torch.rand(4, 4), 0.1, spatial_dims=4)
