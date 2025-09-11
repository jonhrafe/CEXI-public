# CEXI Model Tutorial 

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)

**Cellular EXchange Imaging (CEXI) Model** - Tutorial implementing the CEXI model for diffusion MRI signal modeling with membrane permeability effects.

## Overview

The CEXI model describes water diffusion in biological tissue with **five biophysical parameters**:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Cell radius | R | μm | Average cell size (2-15 μm) |
| Intracellular diffusivity | D<sub>i</sub> | m²/s | Restricted water mobility inside cells |
| Extracellular diffusivity | D<sub>e</sub> | m²/s | Hindered water mobility outside cells |
| Volume fraction | f | - | Intracellular space fraction (0.3-0.85) |
| Membrane permeability | κ | μm/s | Water exchange rate across membranes (1-50 μm/s) |

### How to run

```bash
# Clone the repository
git clone <repository-url>
cd cexi-tutorial

# Install requirements
pip install numpy scipy matplotlib jupyter

# Start the tutorial
jupyter notebook CEXI_Tutorial.ipynb
```

### Files in this Release

- **`CEXI_Tutorial.ipynb`** - Tutorial
- **`utils_inverse_multidelta.py`** - Core CEXI model

## Key Results

### Signal Physics Verification

```
============================================================

1. Sphere signal (restricted diffusion):
   R= 3μm: S(b=2000) = 0.9844  ← Smaller cells
   R= 5μm: S(b=2000) = 0.8947
   R= 8μm: S(b=2000) = 0.5576
   R=12μm: S(b=2000) = 0.1379  ← Larger cells
   ✅ Physics correct: Smaller cells → LESS attenuation

2. Exchange effects with permeability:
   κ=  0 μm/s: S(b=2000) = 0.5877  ← No exchange
   κ=  5 μm/s: S(b=2000) = 0.5767
   κ= 25 μm/s: S(b=2000) = 0.5366
   κ=100 μm/s: S(b=2000) = 0.4366  ← High exchange
   ✅ Exchange effects clearly visible (5-15% changes)
```

### Exchange Time Formula

**Corrected formulation**:
- Intracellular exchange rate: **k_i = (1-f) × 3κ/R**
- Exchange time: **τ_ie = R / (3κ(1-f))**

Example with typical parameters (R=5μm, κ=25μm/s, f=0.65):
```
k_i = (1-0.65) × 3×25/5 = 5.25 s⁻¹
τ_ie = 190 ms
```

## Contents

The interactive Jupyter notebook covers:

### Part 1: Individual Compartment Signals
- **Intracellular**: Restricted diffusion in spheres (non-exponential decay)
- **Extracellular**: Hindered diffusion (exponential decay)
- **Physics verification**: Smaller cells → less signal attenuation

### Part 2: Combined CEXI Signal with Exchange
- **Permeability effects**: How κ affects signal decay
- **Time dependence**: Multi-delta protocols for exchange characterization
- **Exchange time calculation**: Exchange conversion 

### Part 3: Model Fitting
- **Synthetic data generation**: With Rician noise
- **Parameter estimation**: 5-parameter CEXI model fitting
- **Accuracy assessment**: Parameter identifiability analysis

### Part 4: Protocol Optimization
- **Single vs Multi-delta**: Protocol design recommendations
- **Parameter sensitivity**: Which parameters are easier/harder to estimate
- **Best practices**: b-value ranges, timing parameters

## Recommendations

Based on the corrected implementation:

### Acquisition Parameters
- **Diffusion times (Δ)**: 20-80 ms to probe exchange at different timescales
- **Multi-delta**: At least 3 different Δ values for exchange characterization

### Parameter Ranges
- **Permeability (κ)**: Keep < 50 μm/s for stable fitting
- **Volume fraction (f)**: 0.4-0.8 for biological tissues
- **Cell radius (R)**: 2-15 μm depending on tissue type
- **Diffusivities**: D_i < D_e (restricted < hindered)

## Applications

The CEXI model is particularly useful for:

- **Cancer detection**: Membrane permeability changes in tumors
- **Treatment monitoring**: Therapy effects on cell membrane integrity
- **Tissue characterization**: Distinguishing different tissue types
- **Microstructure quantification**: Cell size and packing density

## 📖 Model Comparison

| Aspect | Ball-Sphere Model | CEXI               |
|--------|------------------|---------------------|
| **Membranes** | Impermeable barriers | Permeable with k_i = (1-f)×3κ/R |
| **Parameters** | 4 (R, f, D_i, D_e) | 5 (R, f, D_i, D_e, κ) |
| **Exchange modeling** | None | Kärger model with conservation |
| **Best for** | No/minimal exchange | Moderate permeability tissues |
| **Typical bias** | Overestimates R if exchange exists | Stable for κ < 50 μm/s |
| **Protocol needs** | Single delta sufficient | Multi-delta essential |

## Important Notes

### Stability Warnings
- **High permeability instability**: The model becomes unstable for κ > 50 μm/s
- **Parameter correlations**: Some parameters are harder to estimate than others
- **Protocol dependence**: Multi-delta protocols strongly recommended

### Model Limitations
- **Gaussian assumption**: Assumes Gaussian phase distribution
- **Single cell size**: Uses mean radius (not distribution)
- **Isotropic assumption**: No directional preferences
- **PGSE only**: Developed for pulsed gradient spin echo sequences

### Requirements
- Python 3.7+
- NumPy (array operations)
- SciPy (optimization, special functions)
- Matplotlib (plotting)
- Jupyter (interactive tutorial)

## 📄 Citation

If you use this implementation in your research, please cite:

Gardier R, Villarreal Haro JL, Canales-Rodríguez EJ, Jelescu IO, Girard G, Rafael-Patiño J, Thiran JP. Cellular Exchange Imaging (CEXI): Evaluation of a diffusion model including water exchange in cells using numerical phantoms of permeable spheres. Magn Reson Med. 2023 Oct;90(4):1625-1640. doi: 10.1002/mrm.29720. Epub 2023 Jun 6. PMID: 37279007.

