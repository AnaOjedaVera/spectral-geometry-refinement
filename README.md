
# Spectral Geometry Refinement (SGR)

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Scientific Software](https://img.shields.io/badge/Scientific-Software-8A2BE2)
![Spectral Analysis](https://img.shields.io/badge/Spectral-Analysis-0077B5)
![Clustering](https://img.shields.io/badge/Unsupervised-Clustering-teal)
![Status](https://img.shields.io/badge/Status-In_Development-orange)

**Spectral Geometry Refinement (SGR)** is an unsupervised framework designed to identify and characterize internal spectral structure within existing clusters using multiple representations of spectral geometry.

Originally developed for asteroid spectral taxonomy, SGR examines spectral properties that may remain unresolved in an initial global clustering partition, including absorption features, spectral shape, local extrema, and continuum trends.

## Overview

SGR is a post-clustering refinement method. It does not replace the initial clustering algorithm. Instead, it evaluates individual clusters to determine whether their observations contain additional spectral structure supported by multiple geometric representations.

The framework combines:

- Spectral feature extraction and multi-view representations.
- Gaussian mixture models (GMMs) and Bayesian Information Criterion (BIC) for evaluating candidate subdivisions.
- Agreement analysis between spectral representations.
- Consensus clustering based on observation co-association.
- Hierarchical refinement with minimum cluster-size and acceptance constraints.
- Quantitative and spectral characterization of the resulting subgroups.

Only subdivisions satisfying the specified acceptance criteria are retained. Clusters without sufficient evidence of internal structure preserve their original assignments.

## Methodology

The SGR workflow consists of five main stages:

1. **Spectral feature extraction:** Characterize relevant properties of the spectral curves.
2. **Multi-view analysis:** Construct complementary representations of spectral geometry.
3. **Candidate evaluation:** Identify potential subdivisions using GMMs and model-selection criteria.
4. **Consensus validation:** Evaluate agreement across representations and construct a consensus partition.
5. **Recursive refinement:** Retain supported subdivisions and recursively evaluate eligible subgroups.

The resulting partition preserves the hierarchical relationship between the initial clusters and their refined subgroups.

## Research Application

SGR was developed as part of a doctoral research project on unsupervised asteroid spectral classification.

The methodology was evaluated using synthetic spectral experiments and applied to stellar, mineral, and asteroid spectra.

In the asteroid application, SGR refined three of the eleven groups identified by Dynamic Mixture of Finite Mixtures of Factor Analysers (Dynamic (MF)^2A), producing a partition containing sixteen spectral groups.

The refinement revealed differences associated with absorption features and continuum behavior, providing additional information for the interpretation of asteroid spectral diversity.

## Repository Structure

The repository is being prepared to include:

- The Python implementation of SGR.
- Configuration files and experimental parameters.
- Reproducible examples using synthetic spectra.
- Evaluation and visualization utilities.
- Documentation of the refinement methodology.
- Tests and instructions for reproducing the reported analyses.

## Availability

**Status: In development**

The research implementation is currently being organized for public release. Installation and usage instructions will be provided when the validated code and examples are available.

## Limitations

SGR is designed to identify candidate internal spectral structure, not to establish new physical or taxonomic classes automatically.

Its performance depends on spectral separability, noise, the initial partition, and the geometric representations considered. Subdivision acceptance does not independently establish physical distinctness or reproducibility across datasets.

## Citation

A formal software citation will be added when the repository is released and the associated research reference is finalized.

## Authors and Acknowledgments

Developed as part of doctoral research at the Instituto Politécnico Nacional (IPN), Mexico.

Detailed authorship, institutional acknowledgments, and licensing information will be finalized before public release.
