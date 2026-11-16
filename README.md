I'm a Research Associate at [The Alan Turing Institute](https://www.turing.ac.uk/) in the AI for Physics team, working on physics-informed scalable emulators on the sphere. I completed my PhD in computational cosmology at the [AstroParticule & Cosmology Laboratory (APC, CNRS/IN2P3)](https://apc.in2p3.fr/) in Paris, where I built differentiable and distributed cosmological simulations for the next generation of cosmological surveys.

## Research Focus

- **Emulators on the sphere**: Physics-informed, scalable emulators for weather forecasting
- **Full-field weak lensing inference**: Developing scalable simulation pipelines for LSST
- **CMB component separation**: Part of the [Simons Observatory](https://simonsobservatory.org/) collaboration
- **Cross-survey cosmology**: Combining CMB and large-scale structure datasets for enhanced constraints
- **High-performance computing**: JAX, CUDA, and distributed computing for cosmological modelling

## Main Projects

### Core Packages

| Project | Description |
|---------|-------------|
| [**S2FFT**](https://github.com/astro-informatics/s2fft) | Differentiable and accelerated spherical transforms. Working on highly optimised and distributed CUDA kernels for spherical harmonic transforms, targeting weather forecasting applications |
| [**jax-fli**](https://github.com/DifferentiableUniverseInitiative/jax-fli) | JAX Based Field Level Inference package with fully distributed and high resolution nbody and lensing simulations |
| [**jaxDecomp**](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) | JAX bindings to NVIDIA cuDecomp for distributed 3D FFTs and multi-GPU domain decomposition |
| [**FURAX_CS**](https://github.com/CMBSciPol/furax-cs) | Component separation pipeline for the Simons Observatory and LiteBird using FURAX |
| [**CADRE**](https://github.com/CMBSciPol/CADRE) | Robust GPU-accelerated minimiser for parametric CMB component separation under low signal-to-noise and spatially varying foreground SEDs |
| [**JaxPM**](https://github.com/DifferentiableUniverseInitiative/JaxPM) | Particle-mesh cosmological simulation toolkit in JAX with multi-accelerator support |
| [**FURAX**](https://github.com/CMBSciPol/furax) | Framework for Unified and Robust data Analysis with JAX for inverse problems in cosmology |
| [**jax-healpy**](https://github.com/CMBSciPol/jax-healpy) | JAX-native HEALPix utilities, GPU-ready and differentiable |
| [**jax-grid-search**](https://github.com/CMBSciPol/jax-grid-search) | Distributed grid search + gradient-based optimisation built on JAX/Optax |

## Tech Stack
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=white)
![JAX][jax_badge_link]
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/-C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)

**Specialties:** Distributed Computing | Automatic Differentiation | Bayesian Inference | HPC | CMB Analysis | Weak Lensing

**Website:** [askabalan.github.io](https://askabalan.github.io/)


<!-- Simplified JAX badge. To restore your logo badge, replace the line below with your original base64 `[jax_badge_link]: ...` definition. -->
[jax_badge_link]: https://img.shields.io/badge/-JAX-15c?style=flat-square
