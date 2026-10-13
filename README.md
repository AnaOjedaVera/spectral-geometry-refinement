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

## Run the demonstration (CPU; no external dataset required)

The [synthetic experiment notebook](examples/SGR_Demo.ipynb) is the **recommended first step** for new users. It generates 120 VIS–NIR spectra belonging to four known synthetic populations organized into two initial global clusters. One global cluster contains three underlying populations; the other is a homogeneous control. SGR receives only the initial global labels and spectral measurements, not the known population labels.

From the repository root, install the Python dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab examples/SGR_Demo.ipynb
```

Run the notebook cells in order. It reports the original and refined cluster counts, accepted splits, an adjusted Rand index (ARI) against the known synthetic labels, a **full-spectrum Silhouette score** and the internal consensus diagnostics. The notebook also plots the synthetic and recovered spectral profiles and checks the stopping rule. Refer to [examples/README.md](examples/README.md) for interpretation and details. The example requires **no asteroid dataset and no GPU**.

**Important:** the consensus Silhouette used internally by SGR and the Silhouette calculated in the original spectral feature space are different quantities. Do not interpret a high consensus Silhouette as proof of globally well-separated spectra. The notebook illustrates the algorithm but does not reproduce the dissertation's 16 asteroid groups.

## Run the automated software tests

From the repository root:

```bash
python tests/test_sgr.py
```

The original basic test suite checks (1) the default spectral trend regions, (2) correct handling of clusters too small to refine, including preservation of sample assignments, and (3) rejection of an inconsistent wavelength dimension. It creates its own synthetic inputs; **no input data files are required**.

A successful run ends with `Ran 3 tests` followed by `OK` for this original test suite. If you have replaced it with an expanded test file, the number of tests may differ. `FAIL` means an assertion did not hold, `ERROR` means an unexpected exception occurred, and `Ran 0 tests` means no tests were discovered. See [tests/README.md](tests/README.md) for explanations.

These checks validate limited software behavior; they **do not** establish recovery accuracy in general, CPU–GPU equivalence, or reproduction of the thesis asteroid results.

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

## Asteroid reference configuration

[`configs/asteroids.py`](configs/asteroids.py) provides the **default parameters recorded in the original 2-GPU source code**, including the three spectral trend intervals. It is a configuration reference, **not** a complete independent record of every runtime setting used to obtain the dissertation results. Users should not assume the synthetic notebook's CPU configuration reproduces the asteroid experiment.

## Repository contents

```text
spectral-geometry-refinement/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── requirements.txt
├── requirements-gpu-cu12.txt
├── spectral_geometry_refinement_cpu.py
├── spectral_geometry_refinement_gpu.py
├── configs/
│   └── asteroids.py
├── examples/
│   ├── README.md
│   └── SGR_Demo.ipynb
└── tests/
    ├── README.md
    └── test_sgr.py
```

## Citation

SGR was developed as part of doctoral research at Instituto Politécnico Nacional (IPN), Mexico. Use the repository’s [CITATION.cff](CITATION.cff) file or GitHub’s **Cite this repository** feature for the software attribution currently provided. A version-specific archive and DOI may be added after release metadata and reproducibility checks are finalized.

## License

See [LICENSE](LICENSE) for the repository’s stated MIT terms. Confirm ownership and any institutional or third-party licensing obligations before public release.
