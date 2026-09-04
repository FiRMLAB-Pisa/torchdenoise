# torchdenoise

Classical denoisers for MRI: DeepInverse's, wrapped so that complex data and
stacks of slices, contrasts and frames go straight in, plus a locally low-rank
denoiser that uses the correlation between them.

[![Tests](https://github.com/FiRMLAB-Pisa/torchdenoise/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/torchdenoise/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/torchdenoise/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/torchdenoise)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A denoiser written for natural images takes a real `(batch, channel, height,
width)` tensor. MRI gives complex volumes with any number of axes in front of
the spatial ones — slices, contrasts, echoes, dynamic frames, repetitions — and
a *spatial* denoiser has nothing to say about those axes. It must treat each
independently rather than mistake one for a spatial dimension. That is what
these wrappers add, and all they add.

- **Every leading axis is independent** — slice and time are the same case, so
  neither needs an argument to say which it is. There is a test that a stack
  denoises exactly as its entries do one at a time
- **Complex data goes straight in** — wavelets use DeepInverse's own complex
  transform, which thresholds a coefficient's magnitude and so keeps it whole;
  the denoisers that cannot take complex at all are lifted by DeepInverse's
  `ComplexDenoiserWrapper`. Which route a denoiser takes is stated, because the
  two differ by about a fifth of the signal amplitude
- **A 3D denoiser refuses a plane** rather than silently transforming the wrong
  axes, which is what happens otherwise
- **No memory between calls** — DeepInverse's variation denoisers keep their
  iterates and restart only when the shape changes, so the same input denoises
  differently depending on what preceded it. Inside an iterative reconstruction
  that is a memory nobody asked for; it is reset per call, and `warm_start`
  opts back in

## Quick Start

```bash
pip install torchdenoise
```

```python
import torch
from torchdenoise import TV, Wavelet

volume = torch.rand(8, 20, 256, 256, dtype=torch.complex64)  # frames, slices, y, x

Wavelet()(volume, sigma=0.05)                  # per slice and frame
Wavelet(spatial_dims=3)(volume, sigma=0.05)    # per frame, coupling slices
TV(iterations=50)(volume, sigma=0.05)          # complex through the adapter
```

## Related Works

- **DeepInverse** — <https://deepinv.github.io/>. Every classical denoiser here
  is theirs; this package supplies the shapes and dtypes.
- **BART** — <https://mrirecon.github.io/bart/>, whose locally low-rank
  regularisation is the reference for ours.
- Trzasko J, Manduca A. *Local versus global low-rank promotion in dynamic MRI
  series reconstruction.* Proc ISMRM 2011;19:4371.
