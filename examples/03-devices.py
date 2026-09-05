# %% [markdown]
# # Where it runs, and what it costs
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiRMLAB-Pisa/torchdenoise/blob/main/examples/03-devices.ipynb)
#
# Denoising a volume that does not fit the card is a chunking problem, not a
# denoising one. The dispatch layer measures what is free, splits the leading
# axes, and keeps the data on the host until it is needed -- so the same call
# works on a laptop GPU and on nothing but a CPU.

# [3D denoising demo](https://deepinv.org/auto_examples/optimization/demo_3D_denoising.html),
# on a BrainWeb T1 volume, and adds the parts an MRI volume needs: complex data,
# a stack of slices or contrasts, and a denoiser that treats the contrast axis as
# signal.

# %%
try:
    import torchdenoise  # noqa: F401
except ImportError:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torchdenoise",
            "brainweb-dl",
            "matplotlib",
        ],
        check=True,
    )

# %%
import numpy as np
import torch

import torchdenoise as td

torch.manual_seed(0)


def brain(size=128, slices=16):
    """A BrainWeb T1 volume, or ellipses if brainweb-dl is not installed."""
    try:
        from brainweb_dl import get_mri
    except ImportError:
        rows, columns = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
        disc = ((rows / 0.8) ** 2 + (columns / 0.7) ** 2 < 1).astype(float)
        disc = disc * (1 + 0.3 * np.sin(6 * rows) * np.cos(5 * columns))
        return torch.tensor(np.repeat(disc[None], slices, 0), dtype=torch.float32)
    volume = np.asarray(get_mri(sub_id=4, contrast="T1"), dtype=float)
    middle = volume.shape[0] // 2
    taken = volume[middle - slices // 2 : middle + slices // 2]
    index = np.linspace(0, taken.shape[-1] - 1, size).round().astype(int)
    taken = np.rot90(taken[:, index][:, :, index], 2, axes=(1, 2))
    return torch.tensor(taken.copy() / taken.max(), dtype=torch.float32)


truth = brain()
noise = 0.08
noisy = truth + noise * torch.randn_like(truth)

# A series of contrasts over the same anatomy, which is what LLR denoises and
# what the dispatch layer has to move.
weights = torch.tensor([1.0, 0.75, 0.5, 0.3, 0.15])
series = truth[None] * weights[:, None, None, None]
noisy_series = series + noise * torch.randn_like(series)
print("volume", tuple(truth.shape), " series", tuple(series.shape))

# %% [markdown]
# Host-resident data is sent to whatever CUDA devices there are, a chunk at a time,
# and the answer comes back on the host — so a volume larger than the card still
# denoises. What bounds the footprint is how many blocks are decomposed at once,
# not how much data there is: a batched Hermitian eigendecomposition takes about
# 560 KiB of workspace per matrix on CUDA, and `block_batch_size="auto"` solves for
# the largest batch the card has room for.

# %%
denoiser = td.LLR(spatial_dims=2, block_size=8, device="auto")
print(
    "device:", "cuda" if torch.cuda.is_available() else "cpu (nothing to dispatch to)"
)
print("denoised:", tuple(denoiser(noisy_series, 0.4).shape), "back on the host")

# %% [markdown]
# ## What it costs
#
# Timed on whatever this is running on. The point of the table is the ratio
# between the two columns, not the absolute times: dispatching is worth it when
# the denoiser has enough work to cover moving the data, and the wavelet and
# variational denoisers have far less of it per voxel than LLR does.

# %%
import time


def milliseconds(call, repeats=3):
    call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / repeats


print(f"{'denoiser':<18}{'CPU':>10}{'auto':>10}")
for name, build, data in (
    ("Wavelet, 2D", lambda where: td.Wavelet(spatial_dims=2, device=where), noisy),
    ("TV, 2D", lambda where: td.TV(iterations=20, device=where), noisy),
    (
        "LLR, 2D",
        lambda where: td.LLR(spatial_dims=2, block_size=8, device=where),
        noisy_series,
    ),
):
    on_host = milliseconds(lambda b=build, d=data: b("cpu")(d, noise))
    dispatched = milliseconds(lambda b=build, d=data: b("auto")(d, noise))
    print(f"{name:<18}{on_host:9.0f}ms{dispatched:9.0f}ms")
