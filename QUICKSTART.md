# CEXI Tutorial - Quick Start Guide

### Step 1: Install Requirements
```bash
pip install numpy scipy matplotlib jupyter
```

### Step 2: Launch Tutorial
```bash
jupyter notebook CEXI_Tutorial_PaperAligned.ipynb
```

### Exchange Effects ✅
```
CEXI signal with membrane permeability:
   κ=  0 μm/s: S(b=2000) = 0.5877  ← No exchange
   κ= 25 μm/s: S(b=2000) = 0.5366  ← Medium exchange
   κ=100 μm/s: S(b=2000) = 0.4366  ← High exchange (unstable!)
```
## Key Concepts

1. **Individual Compartments**
   - Intracellular: Restricted diffusion in spheres
   - Extracellular: Hindered diffusion (exponential decay)

2. **Exchange Modeling**
   - Kärger model with proper rate formula
   - Time-dependent exchange effects
   - Multi-delta protocol

3. **Model Fitting**
   - 5-parameter CEXI estimation
   - Noise handling with Rician noise
   - Parameter identifiability analysis

4. **Protocol Optimization**
   - Single vs multi-delta comparison
   - Best practice recommendations

### Issues?
