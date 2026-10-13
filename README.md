# Spectral Geometry Refinement (SGR)

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Scientific Software](https://img.shields.io/badge/Scientific-Software-8A2BE2)
![Spectral Analysis](https://img.shields.io/badge/Spectral-Analysis-0077B5)
![Clustering](https://img.shields.io/badge/Unsupervised-Clustering-teal)
![Status](https://img.shields.io/badge/Status-Research_Release-orange)

**Spectral Geometry Refinement (SGR)** is an unsupervised, post-clustering framework for identifying and characterizing internal spectral structure using complementary representations of spectral geometry. SGR evaluates candidate subdivisions using Gaussian mixture models (GMMs), Bayesian Information Criterion (BIC), agreement between spectral views, co-association, and consensus clustering.

## Research reference implementation

**Important: The asteroid results reported in the doctoral thesis (11 global groups refined into 16 groups) were obtained using the original 2-GPU implementation with its three predefined spectral trend intervals.** The GPU file provided here is an unchanged copy of that implementation, renamed for repository use. The CPU file is the original CPU implementation with **only the default trend configuration changed** to match the GPU reference. All other CPU logic remains unchanged.

| | GPU reference (`spectral_geometry_refinement_gpu.py`) | CPU alternative (`spectral_geometry_refinement_cpu.py`) |
|---|---|---|
| Role | Source code used for the thesis asteroid experiments | Accessible CPU implementation, aligned with the thesis trend configuration |
| Trend intervals (default) | `(0.45, 0.55)`, `(0.55, 2.50)`, `(0.45, 2.50)` µm | Same |
| 10-channel windows (default) | Disabled | Disabled (enabled in the original CPU file) |
| GMM implementation | Custom CuPy-based full-covariance GMM | scikit-learn `GaussianMixture` |
| Default numeric precision | GPU `float32` | NumPy / scikit-learn, generally `float64` |
| Multi-GPU root-cluster parallelism | Supported for two GPUs | Not supported |

**The two implementations are not guaranteed to produce identical partitions.** GMM initialization, numerical precision, optimizer/convergence details, and numerical backend can lead to different candidate assignments and refinements. The CPU alternative is provided for accessibility; it has **not** been demonstrated to reproduce the 16-group thesis result. No validation of numerical equivalence is claimed.

The GPU code also includes an optional CPU backend (`SGRConfig(gpu_backend="cpu")`), which uses scikit-learn for GMM fitting but otherwise follows the GPU file's feature-extraction implementation. This should likewise not be assumed to be identical to the separate CPU file without testing.

**Reproduction boundary:** the thesis result additionally requires the same input spectral matrix, wavelength grid, initial global cluster labels, explicit configuration overrides (if any), software environment, and random seeds. The defaults shown here are those present in the uploaded GPU source file; they do not by themselves establish every parameter used in the final experimental run.

## Installation

CPU:

```bash
python -m pip install -r requirements.txt
```

NVIDIA GPU with a **compatible CUDA 12.x runtime**:

```bash
python -m pip install -r requirements-gpu-cu12.txt
```

The correct CuPy package depends on the system CUDA environment; do not install conflicting CuPy variants. Package versions are not pinned because the original experimental environment has not yet been recorded.

## Quick start

SGR accepts a spectral matrix `X` of shape `(n_spectra, n_channels)`, corresponding wavelength values in µm, and an existing global cluster label for each spectrum. The global labels may come from Dynamic (MF)^2A or another upstream clustering method.

CPU:

```python
from spectral_geometry_refinement_cpu import SGRConfig, run_sgr

config = SGRConfig()
result = run_sgr(X, wavelengths, global_labels, config=config, make_plots=False)
print(result.n_global_clusters, result.n_refined_clusters)
```

GPU (default selects GPUs 0 and 1):

```python
from spectral_geometry_refinement_gpu import SGRConfig, run_sgr

config = SGRConfig(gpu_backend="gpu", gpu_devices=(0, 1))
result = run_sgr(X, wavelengths, global_labels, config=config, make_plots=False)
print(result.n_global_clusters, result.n_refined_clusters)
```

The snippets above assume `X`, `wavelengths`, and `global_labels` are already loaded. The parallel GPU path uses Python multiprocessing with the `spawn` start method; execute from a properly guarded Python script when required by your operating system or interactive environment.

### Optional legacy 10-channel trend configuration

To recover the **original CPU trend settings** (not the settings used for the thesis asteroid result), explicitly override:

```python
config = SGRConfig(
    trend_window_channels=(10,),
    trend_include_full_region=True,
    trend_regions=(),
)
```

This alternative can generate many more trend views, significantly increasing computation time and potentially changing accepted splits.

## Method overview

SGR constructs geometric feature representations from supported spectral extrema and regional trends; selects GMM subdivision candidates using BIC and minimum cluster-size criteria; measures agreement among candidate views using ARI; builds a co-association consensus; and recursively refines clusters that satisfy its acceptance criteria. Clusters without accepted subdivision retain their assignments.

SGR does not automatically establish physically distinct asteroid taxonomic classes. Interpretation requires inspection of spectral morphology and appropriate external evidence. Results depend on input clustering, spectral coverage, noise, and configured feature representations.

## Repository contents

```text
spectral-geometry-refinement/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-gpu-cu12.txt
├── spectral_geometry_refinement_cpu.py
└── spectral_geometry_refinement_gpu.py
```

## Citation

SGR was developed in doctoral research at Instituto Politécnico Nacional (IPN), Mexico. A versioned software citation and, if available, an archived DOI will be added after validation and release metadata are finalized.

## License

See [LICENSE](LICENSE). Confirm code ownership and institutional/third-party licensing permissions before public release.
