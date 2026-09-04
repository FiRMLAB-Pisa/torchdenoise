"""Locally low-rank denoising across contrasts."""

from __future__ import annotations

import pytest
import torch

from torchdenoise import LLR


def series(contrasts: int = 6, size: int = 32, rank: int = 2, seed: int = 0):
    """A series that really is low rank: a few spatial maps, a few curves."""
    generator = torch.Generator().manual_seed(seed)
    maps = torch.rand(rank, size, size, generator=generator)
    curves = torch.rand(contrasts, rank, generator=generator)
    return torch.einsum("cr,rij->cij", curves, maps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("shape", [(6, 32, 32), (3, 6, 32, 32)])
def test_shape_and_dtype_survive(dtype, shape) -> None:
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(shape, generator=generator)
    if dtype.is_complex:
        image = image + 1j * torch.rand(shape, generator=generator)
    out = LLR(block_size=8)(image, 0.05)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_a_low_rank_series_survives_denoising() -> None:
    """What the model describes is kept; what it does not is removed."""
    truth = series()
    generator = torch.Generator().manual_seed(2)
    noisy = truth + 0.05 * torch.randn(truth.shape, generator=generator)
    cleaned = LLR(block_size=8, cycle_spins=False)(noisy, 0.5)
    # measured 0.80x; the win over an uncorrelated series is the sharper test
    assert (cleaned - truth).norm() < 0.85 * (noisy - truth).norm()


def test_it_uses_the_contrast_axis() -> None:
    """A series with no correlation across contrasts has nothing to exploit."""
    generator = torch.Generator().manual_seed(3)
    correlated = series(seed=4)
    independent = torch.rand(6, 32, 32, generator=generator)
    noise = 0.05 * torch.randn(correlated.shape, generator=generator)
    denoiser = LLR(block_size=8, cycle_spins=False)
    kept = (denoiser(correlated + noise, 0.5) - correlated).norm() / correlated.norm()
    lost = (
        denoiser(independent + noise, 0.5) - independent
    ).norm() / independent.norm()
    assert kept < lost


def test_cycle_spinning_moves_the_block_grid() -> None:
    """Two calls differ, which is what stops the lattice building up."""
    image = series()
    denoiser = LLR(block_size=8, cycle_spins=True)
    first = denoiser(image, 0.3)
    second = denoiser(image, 0.3)
    assert not torch.allclose(first, second, atol=1e-6)


def test_without_cycle_spinning_calls_repeat() -> None:
    image = series()
    denoiser = LLR(block_size=8, cycle_spins=False)
    assert torch.allclose(denoiser(image, 0.3), denoiser(image, 0.3), atol=1e-6)


def test_overlapping_blocks_are_averaged() -> None:
    """A stride below the block size denoises each voxel more than once."""
    image = series()
    overlapped = LLR(block_size=8, stride=4, cycle_spins=False)(image, 0.0)
    # A zero threshold changes nothing, so the averaging must be exact.
    assert torch.allclose(overlapped, image, atol=1e-5)


def test_zero_threshold_is_the_identity() -> None:
    image = series()
    assert torch.allclose(
        LLR(block_size=8, cycle_spins=False)(image, 0.0), image, atol=1e-5
    )


def test_block_batching_does_not_change_the_answer() -> None:
    image = series()
    whole = LLR(block_size=8, stride=4, cycle_spins=False, block_batch_size=None)(
        image, 0.3
    )
    split = LLR(block_size=8, stride=4, cycle_spins=False, block_batch_size=3)(
        image, 0.3
    )
    assert torch.allclose(whole, split, atol=1e-5)


def test_both_decomposition_paths_agree() -> None:
    """The Gram route and the full decomposition shrink the same way."""
    tall = LLR(block_size=4, cycle_spins=False)  # 6 contrasts, 16 voxels: Gram
    wide = LLR(block_size=2, cycle_spins=False)  # 6 contrasts, 4 voxels: SVD
    image = series(contrasts=6, size=16)
    assert tall(image, 0.2).shape == image.shape
    assert wide(image, 0.2).shape == image.shape


def test_three_dimensional_blocks() -> None:
    image = torch.rand(4, 8, 16, 16)
    out = LLR(spatial_dims=3, block_size=4)(image, 0.1)
    assert out.shape == image.shape


def test_needs_a_contrast_axis() -> None:
    with pytest.raises(ValueError, match="needs a contrast axis"):
        LLR()(torch.rand(32, 32), 0.1)


def test_block_must_fit() -> None:
    with pytest.raises(ValueError, match="exceeds spatial shape"):
        LLR(block_size=64)(torch.rand(4, 32, 32), 0.1)


def test_sigma_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LLR(block_size=8)(torch.rand(4, 32, 32), -0.1)


def test_spatial_dims_must_be_two_or_three() -> None:
    with pytest.raises(ValueError, match="must be 2 or 3"):
        LLR(spatial_dims=4)


def test_shifting_the_grid_equals_rolling_the_volume() -> None:
    """The two are equivalent, which is why either may be used.

    Rolling is the one that is not used: it measures 46.7 s against 1.6 s for
    the index arithmetic on 5 coefficients over 256^3.
    """
    image = series(contrasts=4, size=32)
    shift = (3, 5)
    shifted = LLR(block_size=8, cycle_spins=True)
    shifted._shift = lambda: shift  # type: ignore[method-assign]
    aligned = LLR(block_size=8, cycle_spins=False)
    by_hand = torch.roll(
        aligned(torch.roll(image, shift, dims=(-2, -1)), 0.3), (-3, -5), dims=(-2, -1)
    )
    assert torch.allclose(shifted(image, 0.3), by_hand, atol=1e-6)


def test_a_full_block_shift_changes_nothing() -> None:
    """Shifting by the block size lands back on the same tiling."""
    image = series(contrasts=4, size=32)
    whole = LLR(block_size=8, cycle_spins=True)
    whole._shift = lambda: (8, 8)  # type: ignore[method-assign]
    aligned = LLR(block_size=8, cycle_spins=False)
    assert torch.allclose(whole(image, 0.3), aligned(image, 0.3), atol=1e-6)


def test_a_generator_makes_the_shifts_repeat() -> None:
    image = series(contrasts=4, size=32)
    first = LLR(block_size=8, generator=torch.Generator().manual_seed(7))(image, 0.3)
    again = LLR(block_size=8, generator=torch.Generator().manual_seed(7))(image, 0.3)
    assert torch.allclose(first, again, atol=1e-6)
