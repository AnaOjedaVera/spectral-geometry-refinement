# SGR — Automated Tests

This directory contains automated tests for the Spectral Geometry Refinement (SGR) CPU implementation.

## Requirements

Install the required dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Running the Tests

From the repository root, execute:

```bash
python -m unittest discover -s tests -p "test_sgr.py" -v
```

The tests generate their own synthetic spectral data. **No external datasets, asteroid spectra, or pretrained models are required.**

## Test Coverage

The basic tests verify:

- The default spectral trend configuration.
- Correct handling of groups below the minimum refinement size.
- Detection of incompatible spectral input dimensions.

A successful test run confirms that these software checks pass. It does not establish numerical equivalence between CPU and GPU implementations or reproduce the asteroid results reported in the dissertation.

## Reference Implementation

The original GPU implementation was used to obtain the dissertation's asteroid refinement results. The tests in this directory currently target the CPU implementation.
