# Spectral Geometry Refinement (SGR) — Software Tests

This directory documents the automated checks for the CPU implementation of Spectral Geometry Refinement (SGR). The checks are intended to identify software regressions, **not** to establish the scientific performance of the clustering method.

## Purpose

The tests validate fundamental behaviors of the implementation: preserving the thesis-reference spectral trend configuration, respecting minimum parent-cluster sizes, preserving sample assignments, and rejecting inconsistent wavelength dimensions. They use small synthetic arrays generated inside the test script. **No external datasets, asteroid spectra, pretrained models, or GPU hardware are required.**

## How to run

Install the core dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

If `test_sgr.py` is **in the repository root**, run:

```bash
python test_sgr.py
```

If `test_sgr.py` is **inside `tests/`**, run:

```bash
python tests/test_sgr.py
```

Both layouts are valid. You do not need to create a `tests/` directory just to execute the script. Run commands from the repository root so that the test script can locate `spectral_geometry_refinement_cpu.py`.

## What the current three tests do

| Test | Input / operation | Expected behavior | Meaning of `OK` |
| --- | --- | --- | --- |
| `test_source_trend_configuration` | Inspect the default `SGRConfig` | Trend windows are disabled; intervals are `(0.45, 0.55)`, `(0.55, 2.50)`, and `(0.45, 2.50)` µm | The CPU defaults match the documented spectral-trend reference settings |
| `test_parent_size_stop_and_conservation` | Create 12 synthetic spectra with 31 channels, initially assigned to two groups of six; set `min_parent_n=80` | 2 initial clusters, 0 accepted splits, 2 refined clusters; all 12 observations retain valid assignments | SGR respects the parent-size stopping rule without losing spectra |
| `test_reject_mismatched_wavelengths` | Provide 31 spectral channels but only 30 wavelength coordinates | Raise `ValueError` | SGR rejects a dimensionally inconsistent input instead of silently proceeding |

These tests **do not need input files**: they construct their own synthetic spectra in memory.

## Understanding the output

A successful execution of the three-test version typically ends with:

```text
...
----------------------------------------------------------------------
Ran 3 tests in ...s

OK
```

The dots indicate tests that passed. `OK` means **all three assertions or expected exceptions succeeded**; it does not mean the recovered spectral clusters are scientifically correct.

Other possible outcomes:

| Output | Meaning | Suggested action |
| --- | --- | --- |
| `OK` | All discovered tests passed | No action needed for these specific checks |
| `FAIL` | A test ran, but an assertion was false | Read the failed test name and compare the observed result with the expected behavior |
| `ERROR` | A test could not complete because of an unexpected exception | Inspect the traceback; check imports, dependencies, file locations, and the code under test |
| `Ran 0 tests` | No tests were discovered | Check the filename, location, and how the command was launched |

The `ValueError` raised deliberately by the third test is **not an error in the test report** when the test finishes as `OK`: the test is explicitly checking that this exception occurs.

## Interpretation and limitations

Passing these tests establishes only that the listed software checks behaved as expected in the current environment. It **does not** demonstrate:

- recovery of the true number of spectral populations;
- high ARI or Silhouette scores;
- detection of genuine absorption-band substructure;
- rejection of all spurious splits;
- numerical equivalence between the CPU and GPU Gaussian-mixture implementations;
- reproducibility of the 16 refined asteroid groups reported in the dissertation.

To evaluate **scientific recovery**, run [`../examples/SGR_Demo.ipynb`](../examples/SGR_Demo.ipynb), which compares SGR results with known synthetic population labels using ARI and computes Silhouette in the full spectral space.

For future software verification, more tests can be added for deterministic synthetic accepted-split scenarios, output conservation after refinement, random-seed sensitivity, and CPU/GPU comparisons. Those checks should be introduced with verified reference outputs rather than assumed expected results.

## Reference implementation

The dissertation's asteroid refinement results were obtained using the original **two-GPU implementation**. The current three tests exercise the **CPU implementation** and do not constitute a reproduction of those dissertation results.
