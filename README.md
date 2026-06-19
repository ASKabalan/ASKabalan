I'm a PhD candidate in computational cosmology at the [AstroParticule & Cosmology Laboratory (APC, CNRS/IN2P3)](https://apc.in2p3.fr/) in Paris, building differentiable and distributed cosmological simulations for the next generation of cosmological surveys.

## Research Focus

- **Full-field weak lensing inference** - Developing scalable simulation pipelines for LSST
- **CMB component separation** - Part of the [Simons Observatory](https://simonsobservatory.org/) collaboration
- **Cross-survey cosmology** - Combining CMB and large-scale structure datasets for enhanced constraints
- **High-performance computing** - Leveraging JAX, CUDA, and distributed computing for cosmological modeling

## Main Projects

### Core Packages

| Project | Description |
|---------|-------------|
| [**jax-fli**](https://github.com/ASKabalan/jax-fli) | JAX Based Field Level Inference package with fully distributed and high resolution nbody and lensing simulations |
| [**jaxDecomp**](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) | JAX bindings to NVIDIA cuDecomp for distributed 3D FFTs and multi-GPU domain decomposition |
| [**FURAX_CS**](https://github.com/CMBSciPol/furax-cs) | Component separation pipeline for the Simons Observatory and LiteBird using FURAX |
| [**CADRE**](https://github.com/CMBSciPol/CADRE) | Robust GPU-accelerated minimizer for parametric CMB component separation under low signal-to-noise and spatially varying foreground SEDs |
| [**JaxPM**](https://github.com/DifferentiableUniverseInitiative/JaxPM) | Particle-mesh cosmological simulation toolkit in JAX with multi-accelerator support |
| [**FURAX**](https://github.com/CMBSciPol/furax) | Framework for Unified and Robust data Analysis with JAX for inverse problems in cosmology |
| [**jax-healpy**](https://github.com/CMBSciPol/jax-healpy) | JAX-native HEALPix utilities - GPU-ready and differentiable |
| [**jax-grid-search**](https://github.com/CMBSciPol/jax-grid-search) | Distributed grid search + gradient-based optimization built on JAX/Optax |

### Contributions

- [**S2FFT**](https://github.com/astro-informatics/s2fft) — Differentiable spherical/Wigner transforms (JAX & PyTorch).
  *Contribution:* translated part of the spherical harmonics algorithm to CUDA to avoid long JAX JIT compile times.

## Tech Stack
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=white)
![JAX][jax_badge_link]
![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/-C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)

**Specialties:** Distributed Computing | Automatic Differentiation | Bayesian Inference | HPC | CMB Analysis | Weak Lensing

**Website:** [askabalan.github.io](https://askabalan.github.io/)


<!-- Simplified JAX badge. To restore your logo badge, replace the line below with your original base64 `[jax_badge_link]: ...` definition. -->
[jax_badge_link]: https://img.shields.io/badge/-JAX-15c?style=flat-square
