# CEXI Model Tutorial

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)

**Cellular EXchange Imaging (CEXI)** — a tutorial implementation of the CEXI model
for diffusion-MRI signal modeling with membrane permeability (water exchange).

## Overview

CEXI describes the diffusion-MRI signal of tissue as two compartments coupled by
water exchange across permeable cell membranes:

- **Intracellular** — water restricted inside spherical cells (radius `R`);
- **Extracellular** — water in the hindered space around them;

with five biophysical parameters:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Cell radius | R | μm | Average cell size (2–15 μm) |
| Intracellular diffusivity | D<sub>i</sub> | m²/s | Mobility inside cells |
| Extracellular diffusivity | D<sub>e</sub> | m²/s | Mobility outside cells |
| Volume fraction | f | – | Intracellular fraction (0.3–0.85) |
| Membrane permeability | κ | μm/s | Water exchange across the membrane |

## The model

For a PGSE measurement with b-value `b`, the signal is a two-compartment Kärger
exchange model,

```
S(b) = f1' · exp(-b · D1') + f2' · exp(-b · D2')
```

where the apparent diffusivities `D1', D2'` and fractions `f1', f2'` come from
exchange between:

- a **restricted-sphere** intracellular compartment (van Gelderen / GPD series,
  radius `R`, intrinsic diffusivity `Di`), and
- a **Gaussian** extracellular compartment (`exp(-b·De)`).

Exchange is set by the membrane permeability and the sphere surface-to-volume
ratio (`3/R`):

```
k_i = 3κ / R          (intracellular efflux rate; residence time τ_i = R/(3κ))
k_e = k_i · f/(1-f)   (extracellular influx, by detailed balance)
```

Example (R = 5 μm, κ = 25 μm/s): `k_i = 3·25/5 = 15 s⁻¹`, `τ_i = 67 ms`.

## How to run

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook CEXI_Tutorial.ipynb
```

## Files

- **`CEXI_Tutorial.ipynb`** — interactive tutorial
- **`utils_inverse_multidelta.py`** — the CEXI model and fitting code

## Example signals

Single shell at `b = 2000 s/mm²` (δ = 10 ms, Δ = 40 ms), `Di = 1e-9`,
`De = 2e-9 m²/s`, `f = 0.65`:

```
Restricted sphere (Di = 1e-9):
   R =  3 μm:  S = 0.9684     ← smaller cells, less attenuation
   R =  5 μm:  S = 0.8492
   R =  8 μm:  S = 0.5992
   R = 12 μm:  S = 0.3879     ← larger cells, more attenuation

Exchange (R = 5 μm):
   κ =   0 μm/s:  S = 0.5584  ← no exchange
   κ =   5 μm/s:  S = 0.5294
   κ =  25 μm/s:  S = 0.4452
   κ = 100 μm/s:  S = 0.3220  ← strong exchange
```

## Tutorial contents

1. **Compartment signals** — restricted (intracellular) vs Gaussian (extracellular).
2. **Exchange** — how κ changes the signal.
3. **Time dependence** — multi-Δ protocols, the exchange signature.
4. **Fitting** — 5-parameter estimation with `fit_CEXI`, parameter recovery.

## Recommendations

- **Multi-Δ acquisition** (≥ 3 diffusion times, Δ ≈ 20–80 ms) is essential to
  separate restriction from exchange.
- **b-values** up to ~3000 s/mm² for adequate contrast.
- `Di < De` (restricted < hindered) for biological tissue.

## Limitations

- Gaussian phase-distribution approximation (PGSE sequences).
- Single mean cell radius (no size distribution).
- Isotropic (no directional preference).

## Citation

Gardier R, Villarreal Haro JL, Canales-Rodríguez EJ, Jelescu IO, Girard G,
Rafael-Patiño J, Thiran JP. *Cellular Exchange Imaging (CEXI): Evaluation of a
diffusion model including water exchange in cells using numerical phantoms of
permeable spheres.* Magn Reson Med. 2023 Oct;90(4):1625-1640.
doi: 10.1002/mrm.29720. PMID: 37279007.
