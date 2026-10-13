# SGR Demo — Controlled Synthetic Spectral Refinement

This folder contains a reproducible Jupyter notebook illustrating how **Spectral Geometry Refinement (SGR)** detects substructure within an existing spectral partition. It uses the **CPU implementation** of SGR and synthetic VIS–NIR reflectance spectra with known population labels.

> **Scope:** This is a demonstration of the software workflow, **not** a reproduction of the dissertation's asteroid results (11 global groups refined into 16). Those results were obtained using the original GPU reference implementation and the corresponding asteroid data and cluster assignments.

## Files

- [`SGR_Demo.ipynb`](SGR_Demo.ipynb) — data generation, SGR refinement, evaluation, plots, and a stopping-rule check.
- [`../spectral_geometry_refinement_cpu.py`](../spectral_geometry_refinement_cpu.py) — CPU implementation imported by the notebook.
- [`../requirements.txt`](../requirements.txt) — Python dependencies.

## Requirements

- Python 3.10 or newer is recommended.
- JupyterLab or Jupyter Notebook to open `.ipynb` files.
- Packages listed in the repository's `requirements.txt`: NumPy, pandas, SciPy, scikit-learn, and Matplotlib.
- **No GPU or CuPy installation is required** for this example.

## Run the notebook

Clone or download the **entire repository**, preserving the `examples/` folder and the CPU module in the repository root. From a terminal in the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab examples/SGR_Demo.ipynb
```

Alternatively, launch JupyterLab in the repository root, open `examples/SGR_Demo.ipynb`, select your Python kernel, then choose **Run → Run All Cells**.

The notebook searches for `spectral_geometry_refinement_cpu.py` in the **current working directory or its parent**. If you use a custom notebook working directory, run Jupyter from the repository root or from `examples/` so the module can be located.

## Experimental design

The notebook generates **120 synthetic spectra** sampled at **61 wavelengths** between **0.45 and 2.45 µm**:

| Initial cluster | Spectra | Synthetic structure | Expected behavior |
| --- | ---: | --- | --- |
| `G1` | 90 | Three hidden populations (30 spectra each) with varying absorption-band locations, depths, and continuum slopes | Recover distinct spectral subgroups if supported by the SGR acceptance criteria |
| `G2` | 30 | One comparatively homogeneous population | Remain a single group |

The known population labels are used **only for evaluation**. SGR receives the spectral matrix, wavelengths, and **two original cluster labels**, not the hidden labels.

The example applies the original reference trend intervals:

```python
trend_window_channels=()
trend_include_full_region=False
trend_regions=((0.45, 0.55), (0.55, 2.50), (0.45, 2.50))
```

The notebook intentionally uses `gmm_n_init=2` to limit demonstration runtime. **This is not the complete experimental configuration of the thesis**, and CPU and GPU GMM implementations are not guaranteed to produce identical partitions.

## Reading the results

The notebook reports:

1. **Accepted splits and refined groups:** how many subdivisions pass SGR's criteria.
2. **Contingency table:** alignment between recovered groups and the known synthetic populations.
3. **Adjusted Rand Index (ARI):** recovery accuracy against synthetic ground truth; `ARI = 1` indicates perfect agreement up to label permutations.
4. **Spectral Silhouette:** separation evaluated directly in the full reflectance feature space.
5. **Consensus Silhouette:** an internal SGR diagnostic computed in the consensus representation; **it is not the same as spectral Silhouette**.
6. **Homogeneous control:** whether the original `G2` cluster remains unsplit.
7. **Stopping-rule check:** verifies that setting `min_parent_n` above both original cluster sizes prevents refinement.

For the bundled synthetic setup, a successful run may recover **four final groups** from the two initial groups: three within `G1` and one unchanged `G2`. Exact results can vary with package versions, initialization, and numerical environment; examine the printed diagnostics rather than assuming particular scores.

## Limitations

- This is a deliberately structured synthetic example, not a comprehensive benchmark or a taxonomic validation.
- High ARI on synthetic populations does not demonstrate performance on real asteroid data.
- A high **consensus** Silhouette does not establish high separation in the complete spectral space.
- No CPU–GPU numerical equivalence is claimed.
- The deterministic stopping-rule check validates one algorithmic constraint; it is not a statistical estimate of false-positive splitting.

For the algorithm's scientific context, configuration details, and citation information, see the [main repository README](../README.md).

