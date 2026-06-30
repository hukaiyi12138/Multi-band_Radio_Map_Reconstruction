# DUSPF-RME

**DUSPF-RME** is a **D**eep **U**nrolled network with **S**LF-**P**SD **F**actorization for **R**adio **M**ap **E**stimation. It is designed for multi-band radio map completion from sparse measurements by integrating physics-informed propagation priors, SLF-PSD factorization, and ADMM-based deep unrolling.

## Overview

DUSPF-RME formulates multi-band radio map estimation as an SLF-PSD-constrained tensor completion problem. Instead of relying on generic tensor nuclear-norm minimization, the proposed method represents the radio map tensor using an emitter-aware low-rank structure. Specifically, each transmitter is modeled by a spatial loss field (SLF) and a power spectral density (PSD) vector, enabling efficient multi-frequency radio map recovery under highly sparse measurements.

The optimization problem is unrolled into a deep network based on ADMM iterations. This combines the interpretability of model-driven optimization with the representation power of deep neural networks.

## Key Features

### SLF-PSD Factorization

DUSPF-RME decomposes the multi-band radio map into a rank-(R) outer-product representation between per-emitter SLFs and PSD vectors. This structure captures frequency-domain low-rank correlations while avoiding singular value thresholding operations required by Tucker nuclear-norm minimization.

### Joint PSD Initialization

The PSD coefficients are initialized using a Tikhonov-regularized joint least-squares procedure. This initialization exploits inter-emitter coupling on the observed support and provides a stable starting point for subsequent ADMM iterations.

### Physics-Informed Propagation Prior

For each transmitter, DUSPF-RME constructs a learnable physics-informed propagation prior by combining free-space path loss with ray-cast geometric shadowing. The path-loss exponent and shadowing attenuation coefficient are optimized end-to-end with the network. Since the prior depends only on the building map and transmitter locations, it remains informative even when the available measurements are extremely sparse.

### Physics-Aware U-Net Proximal Operator

The SLF proximal mapping is implemented using a learned U-Net module. The U-Net refines the SLF estimate by predicting a residual correction on top of the propagation prior. It is conditioned on physically meaningful per-emitter features, including the distance map, ray-cast shadowing field, building map, and propagation prior.

Compared with shallow local convolutional modules, the U-Net provides a larger effective receptive field, which helps propagate sparse observations across the spatial grid under extremely low sampling rates.

## Experimental Results

Experiments on the BART-Lab Radiomap dataset show that DUSPF-RME consistently outperforms existing methods across fiber sampling rates from 1% to 15%. Compared with RadioUNet, DUSPF-RME achieves improved PSNR and outage detection accuracy, demonstrating its effectiveness in sparse multi-band radio map estimation.
