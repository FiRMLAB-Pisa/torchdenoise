"""Locally low-rank denoising.

A spatial denoiser treats contrasts and frames as axes it has nothing to say
about. This one is the opposite: it is the correlation *between* them that it
uses. Within a small spatial block, the signal across contrasts is nearly the
same curve scaled, so the block's matrix -- one row per contrast, one column
per voxel -- is close to low rank, while noise is not. Shrinking its singular
values removes what does not fit that description.

Blocks overlap, so a voxel is denoised several times and the results averaged,
and the block grid is shifted between calls. Without one or the other the block
edges show as a lattice.

The blocking and cycle-spinning follow SetsompopLab/MRF, BSD-3-Clause; see
NOTICE_LLR.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ._dispatch import USABLE_FRACTION, available_devices, free_bytes, run_chunked

__all__ = ["LLR"]

# What a batched Hermitian eigendecomposition costs on CUDA, per matrix. It is
# measured, and it is the whole footprint: the block extraction that looks
# expensive is 0.08 GiB where the decomposition of the same blocks is 8.57, and
# the cost per matrix is the same for a 5x5 as for an 8x8, so it is workspace
# rather than arithmetic. Taking the singular values directly instead is 60
# times lighter and 1700 times slower, which is no escape; the batch is bounded
# instead.
EIGH_WORKSPACE_PER_MATRIX = 560 << 10


def _per_axis(
    value: int | Sequence[int], dimension: int, *, name: str
) -> tuple[int, ...]:
    """One positive integer per spatial axis."""
    result = (
        (value,) * dimension if isinstance(value, int) else tuple(int(v) for v in value)
    )
    if len(result) != dimension or any(item < 1 for item in result):
        raise ValueError(f"{name} must contain {dimension} positive integers")
    return result


class LLR(torch.nn.Module):
    """Shrink the singular values of small blocks taken across contrasts.

    Parameters
    ----------
    spatial_dims
        2 or 3. The axis just before the spatial ones is the low-rank axis --
        contrasts, frames, echoes, subspace coefficients -- and anything before
        *that* is a batch of independent problems.
    block_size, stride
        Block extent and the step between block origins, per spatial axis.
        ``stride`` defaults to ``block_size``, which tiles without overlap.
    cycle_spins
        Shift the block grid by one voxel per call, so the lattice a fixed grid
        leaves does not survive several passes of an iterative reconstruction.
    block_batch_size
        How many blocks are decomposed at once. This is the footprint: block
        extraction materialises ``entries x blocks x contrasts x block_voxels``,
        which turns 40 MiB of data into gigabytes of workspace. ``"auto"``
        solves for the largest batch the device has room for, which is also the
        fastest; ``None`` does every block at once regardless.
    device
        Where to run. ``"auto"`` uses every visible CUDA device for
        host-resident data, dealing chunks out between them, and leaves the
        work where it is when there is no card.

    Examples
    --------
    >>> import torch
    >>> from torchdenoise import LLR
    >>> series = torch.rand(5, 32, 32, dtype=torch.complex64)  # contrasts, y, x
    >>> LLR(block_size=8)(series, 0.05).shape
    torch.Size([5, 32, 32])
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 2,
        block_size: int | Sequence[int] = 8,
        stride: int | Sequence[int] | None = None,
        cycle_spins: bool = True,
        block_batch_size: int | str | None = "auto",
        device: str | torch.device | None = "auto",
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        if block_batch_size not in (None, "auto") and (
            not isinstance(block_batch_size, int)
            or isinstance(block_batch_size, bool)
            or block_batch_size < 1
        ):
            raise ValueError(
                "block_batch_size must be a positive integer, None, or 'auto'"
            )
        self.spatial_dims = spatial_dims
        self.block_size = _per_axis(block_size, spatial_dims, name="block_size")
        self.stride = (
            self.block_size
            if stride is None
            else _per_axis(stride, spatial_dims, name="stride")
        )
        self.cycle_spins = cycle_spins
        self.block_batch_size = block_batch_size
        self.device = device
        self._calls = 0
        self._key: tuple[object, ...] | None = None
        self._grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _coordinates(
        self, spatial: tuple[int, ...], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Block origins, in-block offsets, and the strides that flatten them."""
        key = (*spatial, str(device))
        if key == self._key and self._grid is not None:
            return self._grid

        starts = []
        for size, block, step in zip(
            spatial, self.block_size, self.stride, strict=True
        ):
            if block > size:
                raise ValueError(
                    f"block_size {self.block_size!r} exceeds spatial shape {spatial!r}"
                )
            origins = list(range(0, size - block + 1, step))
            if origins[-1] != size - block:
                origins.append(size - block)
            starts.append(torch.tensor(origins, dtype=torch.long, device=device))

        origin_grid = torch.meshgrid(*starts, indexing="ij")
        offset_grid = torch.meshgrid(
            *[
                torch.arange(b, dtype=torch.long, device=device)
                for b in self.block_size
            ],
            indexing="ij",
        )
        flat_strides = torch.tensor(
            [
                int(torch.tensor(spatial[index + 1 :]).prod().item())
                if index + 1 < self.spatial_dims
                else 1
                for index in range(self.spatial_dims)
            ],
            dtype=torch.long,
            device=device,
        )
        self._key = key
        self._grid = (
            torch.stack([axis.reshape(-1) for axis in origin_grid], dim=-1),
            torch.stack([axis.reshape(-1) for axis in offset_grid], dim=-1),
            flat_strides,
        )
        return self._grid

    def _batch_of_blocks(
        self,
        block_count: int,
        entries: int,
        contrasts: int,
        element: int,
        device: torch.device,
    ) -> int:
        """Return the largest number of blocks that will fit, which is also fastest.

        Block extraction is what costs: ``entries x blocks x contrasts x
        block_voxels``, and about four such tensors are live at once between the
        gather, the permuted copy and the decomposition's workspace.
        """
        if self.block_batch_size != "auto":
            return self.block_batch_size or block_count
        if device.type != "cuda":
            return block_count
        del contrasts, element  # the workspace does not depend on either
        matrices = USABLE_FRACTION * free_bytes(device) / EIGH_WORKSPACE_PER_MATRIX
        return max(1, min(block_count, int(matrices / max(entries, 1))))

    def _bytes_per_entry(self, voxels: int, contrasts: int, element: int) -> int:
        """Return the data one entry of the leading axis needs on the device.

        Only the data: the decomposition's workspace is bounded separately, by
        how many blocks are decomposed at once, so it does not constrain how
        many entries a chunk may hold.
        """
        return contrasts * voxels * element * 4

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """Denoise a series of any shape.

        Parameters
        ----------
        x
            ``(*independent, contrast, *spatial)``, real or complex.
        sigma
            The singular-value threshold: a scalar, or one per independent
            entry.

        Returns
        -------
        torch.Tensor
            The denoised series, in the shape and dtype it arrived in.
        """
        needed = self.spatial_dims + 1
        if x.ndim < needed:
            raise ValueError(
                f"a {self.spatial_dims}D locally low-rank denoiser needs a contrast "
                f"axis and {self.spatial_dims} spatial ones, got shape {tuple(x.shape)}"
            )
        original = x.shape
        spatial = tuple(int(size) for size in original[-self.spatial_dims :])
        contrasts = int(original[-needed])
        folded = x.reshape(-1, contrasts, *spatial)
        entries = folded.shape[0]

        real_dtype = x.real.dtype
        threshold = torch.as_tensor(sigma, dtype=real_dtype, device=x.device).squeeze()
        if threshold.ndim == 0:
            threshold = threshold.expand(entries)
        elif threshold.ndim != 1 or threshold.shape[0] != entries:
            raise ValueError("sigma must be a scalar or hold one value per entry")
        if bool(torch.any(threshold < 0)):
            raise ValueError("sigma must be non-negative")

        # The block grid moves once per call, not once per chunk, or chunks
        # would be denoised on grids offset from one another.
        shifts = tuple(self._calls % block for block in self.block_size)
        self._calls += 1

        devices = available_devices(self.device) if folded.device.type == "cpu" else []
        if not devices or devices[0].type != "cuda":
            return self._denoise(folded, threshold, shifts).reshape(original)

        voxels = 1
        for size in spatial:
            voxels *= size

        def work(chunk: torch.Tensor, device: torch.device, first: int, last: int):
            return self._denoise(chunk, threshold[first:last].to(device), shifts)

        denoised = run_chunked(
            work,
            folded,
            devices=devices,
            bytes_per_entry=self._bytes_per_entry(
                voxels, contrasts, folded.element_size()
            ),
        )
        return denoised.reshape(original)

    def _denoise(
        self, x: torch.Tensor, threshold: torch.Tensor, shifts: tuple[int, ...]
    ) -> torch.Tensor:
        """Shrink every block of a folded, co-located batch."""
        batch, contrasts = x.shape[0], x.shape[1]
        spatial = tuple(int(size) for size in x.shape[2:])
        origins, offsets, flat_strides = self._coordinates(spatial, x.device)
        real_dtype = x.real.dtype

        flat_input = x.reshape(batch, contrasts, -1)
        output = torch.zeros_like(flat_input)
        tiled = all(
            step == block and size % block == 0
            for size, block, step in zip(
                spatial, self.block_size, self.stride, strict=True
            )
        )
        weights = (
            None
            if tiled
            else torch.zeros(output.shape[-1], dtype=real_dtype, device=x.device)
        )
        count = origins.shape[0]
        per_pass = self._batch_of_blocks(
            count, batch, contrasts, x.element_size(), x.device
        )
        shifted = self.cycle_spins and any(shifts)
        if shifted:
            shift = torch.tensor(shifts, dtype=torch.long, device=x.device)
            extent = torch.tensor(spatial, dtype=torch.long, device=x.device)
        else:
            offset_indices = (offsets * flat_strides).sum(dim=-1)

        for first in range(0, count, per_pass):
            taken = origins[first : first + per_pass]
            if shifted:
                where = (taken[:, None, :] + offsets[None, :, :] - shift) % extent
                indices = (where * flat_strides).sum(dim=-1)
            else:
                indices = (taken * flat_strides).sum(dim=-1)[:, None] + offset_indices[
                    None, :
                ]
            matrices = flat_input[:, :, indices].permute(0, 2, 1, 3)
            matrices = _shrink(matrices, threshold, contrasts, real_dtype)

            values = matrices.permute(0, 2, 1, 3).reshape(batch, contrasts, -1)
            flat_indices = indices.reshape(-1)
            spread = flat_indices[None, None, :].expand_as(values)
            if tiled:
                output.scatter_(2, spread, values)
            else:
                output.scatter_add_(2, spread, values)
                weights.scatter_add_(  # type: ignore[union-attr]
                    0, flat_indices, torch.ones_like(flat_indices, dtype=real_dtype)
                )

        if weights is not None:
            output = output / weights.clamp_min(torch.finfo(real_dtype).eps)[None, None]
        return output.reshape(batch, contrasts, *spatial)


def _shrink(
    matrices: torch.Tensor,
    threshold: torch.Tensor,
    contrasts: int,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Soft-threshold the singular values of every block at once.

    With fewer contrasts than voxels in a block the Gram matrix is the smaller
    of the two, and an eigendecomposition of it is cheaper than a singular
    value decomposition of the block. The shrinkage is the same either way.
    """
    smallest = torch.finfo(real_dtype).eps
    if contrasts <= matrices.shape[-1]:
        gram = matrices @ matrices.mH
        eigenvalues, basis = torch.linalg.eigh(gram)
        singular = torch.sqrt(torch.clamp(eigenvalues, min=0))
        factors = torch.where(
            singular > smallest,
            torch.clamp(
                1 - threshold[:, None, None] / singular.clamp_min(smallest), min=0
            ),
            torch.zeros_like(singular),
        )
        return basis @ (factors.unsqueeze(-1) * (basis.mH @ matrices))
    basis, singular, rows = torch.linalg.svd(matrices, full_matrices=False)
    shrunk = torch.clamp(singular - threshold[:, None, None], min=0)
    return (basis * shrunk.unsqueeze(-2)) @ rows
