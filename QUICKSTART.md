# CEXI Tutorial — Quick Start

### 1. Install requirements
```bash
pip install numpy scipy matplotlib jupyter
```

### 2. Launch the tutorial
```bash
jupyter notebook CEXI_Tutorial.ipynb
```

### Exchange effect
CEXI signal at `b = 2000 s/mm²` (R = 5 μm, Di = 1e-9, De = 2e-9 m²/s, f = 0.65):
```
   κ =   0 μm/s:  S = 0.5584   ← no exchange
   κ =  25 μm/s:  S = 0.4452   ← moderate exchange
   κ = 100 μm/s:  S = 0.3220   ← strong exchange
```

## Key concepts

1. **Compartments** — intracellular (restricted diffusion in spheres) and
   extracellular (Gaussian / hindered diffusion).
2. **Exchange** — Kärger model, `k_i = 3κ/R`, with multi-Δ acquisitions to probe
   the time dependence.
3. **Fitting** — 5-parameter estimation (`R, Di, De, f, κ`) with Rician-noise
   handling.
