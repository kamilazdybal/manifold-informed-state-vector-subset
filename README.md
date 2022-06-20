# Manifold-informed state vector subset for reduced-order modeling

This repository contains code and materials to reproduce the results from the "*Manifold-informed state vector subset for reduced-order modeling*" paper.

> K. Zdybał, J. C. Sutherland, A. Parente - *Manifold-informed state vector subset for reduced-order modeling*, 2022, Proceedings of the Combustion Institute

### Our hypothesis

The adequate choice of variables for PCA can have beneficial effects on the low-dimensional manifold topology.

### Our methodology

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-2.png" width="700">
</p>

## Reproducing paper results using Jupyter notebooks

All code used to produced the results in the original publication and in the supplementary material can be found in the Jupyter notebooks provided in the [`code`](code/) directory. [`PCAfold`](https://pcafold.readthedocs.io/en/latest/index.html) library is required.

Below, are the detailed guidelines on reproducing each figure from the original publication:

### 📄 **Figure 1**

This [Jupyter notebook](code/paper-Figure-1-X-vs-XS-demo.ipynb) can be used to generate **Figure 1**:

<p align="center">
  <img src="https://github.com/kamilazdybal/manifold-informed-state-vector-subset/raw/main/figures/Figure-1.png" width="300">
</p>

### 📄 **Figure 2**

This [Jupyter notebook](code/paper-Figure-2-backward-variable-selection-algorithm-explanation.ipynb) can be used to generate the middle frame in **Figure 2**:

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



### 📄 **Figure 6**

This [Jupyter notebook](code/paper-Figure-6-scalings-and-subsetting-ranking-across-dimensionality.ipynb) can be used to generate **Figure 6**:



### 📄 **Figure 7**

This [Jupyter notebook](code/paper-Figure-7-regression-correlation-SYNGAS.ipynb) can be used to generate **Figure 7**:



### 📄 **Figure 8**

This [Jupyter notebook](code/paper-Figure-8-kernel-regression-of-all-variables.ipynb) can be used to generate **Figure 8**:


