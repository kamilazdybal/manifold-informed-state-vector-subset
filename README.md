This repository is licensed under: [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# Manifold-informed state vector subset for reduced-order modeling

This repository contains code and materials to reproduce the results from the "*Manifold-informed state vector subset for reduced-order modeling*" paper.

> [K. Zdybał, J. C. Sutherland, A. Parente - *Manifold-informed state vector subset for reduced-order modeling*, 2022, Proceedings of the Combustion Institute](https://www.sciencedirect.com/science/article/pii/S1540748922000153)

**This paper has received the [Distinguished Paper Award](https://www.combustioninstitute.org/resources/awards/distinguished-papers/) from The Combustion Institute.**

### Our hypothesis

The adequate choice of variables for PCA can have beneficial effects on the low-dimensional manifold topology.

### Our methodology

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-2.png" width="900">
</p>

## Data availability

All datasets used in the current work are provided in the [`data-sets`](data-sets/) directory. The datasets have been generated with the open-source [Spifire](https://spitfire.readthedocs.io/en/latest/) Python library.

## Reproducing paper results using Jupyter notebooks

All code used to produce the results in the original publication and in the supplementary material can be found in the Jupyter notebooks provided in the [`code`](code/) directory. [PCAfold](https://pcafold.readthedocs.io/en/latest/index.html) library is required.

Below, are the detailed guidelines on reproducing each figure from the original publication:

### 📄 **Figure 1**

This [Jupyter notebook](code/paper-Figure-1-X-vs-XS-demo.ipynb) can be used to generate **Figure 1**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-1.png" width="300">
</p>

### 📄 **Figure 2**

This [Jupyter notebook](code/paper-Figure-2-manifold-informed-backward-elimination-algorithm.ipynb) can be used to generate the middle frame in **Figure 2**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-2.png" width="700">
</p>

### 📄 **Figure 3**

This [Jupyter notebook](code/paper-Figure-3-choice-of-target-variables-demo.ipynb) can be used to generate **Figure 3**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-3.png" width="300">
</p>

### 📄 **Figure 4**

This [Jupyter notebook](code/paper-Figure-4-selecting-minor-species-demo.ipynb) can be used to generate **Figure 4**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-4.png" width="300">
</p>

### 📄 **Figure 5**

This [Jupyter notebook](code/paper-Figure-5-scalings-and-subsetting-ranking.ipynb) can be used to generate **Figure 5**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-5.png" width="700">
</p>

### 📄 **Figure 6**

This [Jupyter notebook](code/paper-Figure-6-scaling-and-subsetting-ranking-across-dimensionality.ipynb) can be used to generate **Figure 6**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-6.png" width="700">
</p>

### 📄 **Figure 7**

This [Jupyter notebook](code/paper-Figure-7-regression-correlation-SYNGAS.ipynb) can be used to generate **Figure 7**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-7.png" width="300">
</p>

### 📄 **Figure 8**

This [Jupyter notebook](code/paper-Figure-8-kernel-regression-of-all-variables.ipynb) can be used to generate **Figure 8** (and the analogous supplementary figures):

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-8.png" width="300">
</p>
