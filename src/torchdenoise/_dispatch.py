"""Run a denoiser on the GPUs there are, over data that need not fit on them.

Two separate problems, and only one of them is the transfer.

The first is the working set. Extracting overlapping blocks materialises
``entries x blocks x contrasts x block_voxels``, so 40 MiB of data becomes 8.6
GiB of workspace, and once that exceeds the card the driver starts satisfying
it from host memory and the whole thing runs thirty times slower than it should.
The batch size is therefore chosen from what is actually free rather than
guessed, and measured with ``mem_get_info``: what a process has allocated says
nothing about what the card has left.

The second is the data. When it does not fit at all it is sent a chunk at a
time, and when there is more than one card the chunks go to all of them.
Launches are asynchronous, so issuing to each device in turn and synchronising
at the end has them working at once without a thread in sight.

Overlapping a chunk's transfer with the previous chunk's compute is *not* done.
It was built for the least-squares solver in ``torchsolve``, measured, and
removed: it lost to plain sequential chunking at every size, by 0.13x to 0.79x
from pageable memory and still 0.27x to 0.91x from pinned memory where no
staging copy is needed. It is not assumed to be different here; it would have
to be measured to earn a place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

__all__ = ["available_devices", "free_bytes", "run_chunked"]

# Of what a card reports free: the rest is fragmentation, the allocator's own
# bookkeeping, and the fact that a working set is never exactly what was
# estimated.
USABLE_FRACTION = 0.6


def available_devices(
    requested: str | torch.device | Sequence | None,
) -> list[torch.device]:
    """Which devices to use.

    ``"auto"`` is every visible CUDA device, or none if there are none, in
    which case the work stays where it is.
    """
    if requested is None:
        return []
    if isinstance(requested, (str, torch.device)) and str(requested) != "auto":
        return [torch.device(requested)]
    if not isinstance(requested, (str, torch.device)):
        return [torch.device(item) for item in requested]
    if not torch.cuda.is_available():
        return []
    return [torch.device("cuda", index) for index in range(torch.cuda.device_count())]


def free_bytes(device: torch.device) -> int:
    """How much the card has left, not how much this process has taken."""
    if device.type != "cuda":
        return 0
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)


def run_chunked(
    work: Callable[[torch.Tensor, torch.device, int, int], torch.Tensor],
    x: torch.Tensor,
    *,
    devices: Sequence[torch.device],
    bytes_per_entry: int,
) -> torch.Tensor:
    """Apply ``work`` to ``x`` along its first axis, across the given devices.

    Parameters
    ----------
    work
        Called with a chunk already on a device, that device, and the half-open
        range of first-axis entries the chunk covers, so anything given per
        entry can be sliced to match.
    x
        Host tensor whose first axis is a batch of independent entries.
    devices
        Where to run. Chunks are dealt out in turn, so every card is busy.
    bytes_per_entry
        Working set one entry of the first axis needs, used to size a chunk.

    Returns
    -------
    torch.Tensor
        The result, back on the host, in the shape ``work`` returns per chunk.
    """
    total = x.shape[0]
    spans: list[tuple[int, int, torch.device]] = []
    start = 0
    while start < total:
        device = devices[len(spans) % len(devices)]
        room = int(USABLE_FRACTION * free_bytes(device) / max(bytes_per_entry, 1))
        size = max(1, min(room, total - start))
        spans.append((start, start + size, device))
        start += size

    # Launches are asynchronous, so every device is given its work before any
    # of it is waited on.
    results: list[torch.Tensor | None] = [None] * len(spans)
    for index, (first, last, device) in enumerate(spans):
        with torch.cuda.device(device):
            results[index] = work(
                x[first:last].to(device, non_blocking=True), device, first, last
            )
    for device in {span[2] for span in spans}:
        torch.cuda.synchronize(device)

    return torch.cat([piece.to("cpu") for piece in results])  # type: ignore[union-attr]
