# SGR — Automated Tests

This directory contains lightweight CPU unit tests for Spectral Geometry Refinement (SGR). The tests verify input validation, source-default trend-region configuration, spectrum/label preservation when refinement is not permitted, wavelength ordering, missing-value handling, configuration safeguards, and repeatability in a no-split case.

## Requirements

Run the following commands **from the repository root**:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

A successful run finishes with `OK`.

## What these tests cover

- Reference trend configuration: `(0.45, 0.55)`, `(0.55, 2.50)`, and `(0.45, 2.50)` µm.
- Preservation of all observations when groups are too small for refinement.
- Stable labels when there is no eligible split.
- Rejection of mismatched labels, mismatched or duplicate wavelengths, non-finite wavelengths, and empty spectral channels.
- Sorting of wavelength channels and imputation of isolated missing spectral values.
- Rejection of invalid minimum cluster sizes and invalid trend boundaries.
- Availability of the optional legacy 10-channel trend window.

## Scope and limitations

These are **CPU smoke/unit tests**, not a numerical validation of GPU vs. CPU implementations. In particular, the repeatability test exercises an intentionally no-split case: it does **not** establish repeatability of accepted subdivisions. The tests also do **not** reproduce the thesis's 16 asteroid groups or measure recovery of known synthetic populations. See `examples/SGR_Demo.ipynb` for an independent synthetic demonstration.

For a thesis-reproduction study, additionally retain the original GPU code, spectral inputs, global labels, full experimental configuration, software/CUDA versions, and random seeds. An end-to-end regression test on accepted splits and GPU parity should only be added once reference outputs are archived and reproducibly verified.
