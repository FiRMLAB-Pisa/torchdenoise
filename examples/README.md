# Examples

One example per subject. The `.py` is the source: it runs as a script, lints
with the rest of the package, and reads as a diff. The `.ipynb` beside it is
generated from it, executed, and committed with its outputs, so it opens in
Colab and runs top to bottom on a BrainWeb volume — or on ellipses, if
`brainweb-dl` is not installed.

| example | shows | checked against |
|---|---|---|
| [`01-shapes`](01-shapes.ipynb) | `Wavelet`, `TV`, complex input, `spatial_dims` | a stack denoising exactly as its entries do one at a time |
| [`02-llr`](02-llr.ipynb) | `LLR` across contrasts | a purely spatial denoiser on the same series |
| [`03-devices`](03-devices.ipynb) | `device="auto"`, `block_batch_size` | the same call on the host, and the clock |

[`figures/llr_delics.png`](figures/) is the README's figure and comes from
Deli-CS data that is not in this repository.

## Rebuilding

```bash
pip install -e .[dev] jupytext nbclient ipykernel
bash scripts/build_examples.sh
```

Every notebook is regenerated from its script and executed against the
interpreter the package is installed into. `--check` verifies the notebooks are
current without running them.
