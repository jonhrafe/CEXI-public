#!/usr/bin/env python3
"""
CEXI (Cellular Exchange Imaging): two-compartment permeable-sphere model for the
diffusion-MRI signal, with full multi-delta / multi-timing support (each b-value
can have its own delta, tdiff, G).

Model
-----
1. Intracellular compartment: restricted diffusion inside a sphere of radius R,
   computed with the van Gelderen / Murday-Cotts Gaussian-Phase-Distribution
   series (reflecting-boundary eigenvalues, the zeros of j'_1).
2. Extracellular compartment: Gaussian (hindered) diffusion with diffusivity De.
3. Water exchange across the membrane via the Kärger model, with
   k_i = 3*kappa/R and k_e = k_i*f/(1-f) (detailed balance).

Parameters: R (um), Di, De (m^2/s), f (intracellular fraction), kappa (um/s).
"""

import numpy as np
from scipy.optimize import minimize
import warnings

# ============================================================================
# CONSTANTS
# ============================================================================

# Physical constants
GYRO = 267.5153151e6  # rad/s/T for protons
GYRO_MS = GYRO / 1e3  # rad/ms/T
PI = np.pi
PI_2 = 2 * PI
EPS = 1e-10

# Unit conversions
S_TO_MS = 1e3
MUM_TO_M = 1e-6

# Model limits
PERM_MIN = 0.01  # Minimum permeability in μm/s
PERM_MAX_STABLE = 50.0  # μm/s; above this, exchange estimation is less stable

# ============================================================================
# BESSEL FUNCTION ROOTS FOR SPHERE MODEL
# ============================================================================

def get_bessel_roots_sphere(n_max=32):
    """
    Get first n_max positive roots of  j'_1(x) = 0, where j_1 is the spherical
    Bessel function of order 1.

    These satisfy the REFLECTING (no-flux, Neumann) boundary condition at the
    surface of an impermeable sphere and are the correct eigenvalues for the
    van Gelderen / Murday-Cotts restricted-diffusion GPD expression.

    j'_1(x) = 0  <=>  (x^2 - 2) sin(x) + 2 x cos(x) = 0   (x > 0)

    The first roots are 2.0816, 5.9404, 9.2058, ...

    Reference: van Gelderen et al., JMR B 103 (1994) 255-260; Callaghan,
    "Principles of Nuclear Magnetic Resonance Microscopy".
    """
    from scipy.optimize import brentq

    # j'_1(x) up to a positive prefactor that never vanishes for x > 0
    g = lambda x: (x ** 2 - 2.0) * np.sin(x) + 2.0 * x * np.cos(x)

    # Roots are spaced by ~pi; scan a grid and bracket every sign change.
    x = np.linspace(1e-4, (n_max + 2) * np.pi, 20000)
    gv = g(x)
    roots = []
    for i in range(len(x) - 1):
        if gv[i] * gv[i + 1] < 0:
            roots.append(brentq(g, x[i], x[i + 1]))
        if len(roots) >= n_max:
            break
    return np.array(roots[:n_max])

# Get roots for use in GPD model
BES_ROOTS = get_bessel_roots_sphere(32)

# ============================================================================
# PROTOCOL CLASS - ENHANCED FOR MULTI-DELTA
# ============================================================================

class Protocol:
    """
    Enhanced DWI acquisition protocol with per-measurement parameters.
    Each measurement can have unique (b, delta, tdiff, G) values.
    """
    
    def __init__(self, bvals=None, delta=None, tdiff=None, TE=None, 
                 gdir=None, G=None):
        """
        Initialize protocol with support for varying parameters per measurement.
        
        Parameters:
        -----------
        bvals : array (N,)
            b-values in s/mm²
        delta : float or array (N,)
            Gradient pulse duration(s) in seconds
        tdiff : float or array (N,)  
            Diffusion time (Δ) in seconds
        TE : float or array (N,)
            Echo time in seconds
        gdir : array (N, 3)
            Gradient directions
        G : float or array (N,)
            Gradient strengths in T/m
        """
        # Store b-values
        self.b = np.asarray(bvals) if bvals is not None else np.array([])
        n_meas = len(self.b)
        
        # Handle delta - convert to array if scalar
        if delta is not None:
            if np.isscalar(delta):
                self.delta = np.full(n_meas, delta)
            else:
                self.delta = np.asarray(delta)
                assert len(self.delta) == n_meas, f"delta length {len(self.delta)} != b length {n_meas}"
        else:
            self.delta = np.full(n_meas, 0.005)  # Default 5ms
        
        # Handle tdiff - convert to array if scalar  
        if tdiff is not None:
            if np.isscalar(tdiff):
                self.tdiff = np.full(n_meas, tdiff)
            else:
                self.tdiff = np.asarray(tdiff)
                assert len(self.tdiff) == n_meas, f"tdiff length {len(self.tdiff)} != b length {n_meas}"
        else:
            self.tdiff = np.full(n_meas, 0.020)  # Default 20ms
        
        # Handle TE - convert to array if scalar
        if TE is not None:
            if np.isscalar(TE):
                self.TE = np.full(n_meas, TE)
            else:
                self.TE = np.asarray(TE)
        else:
            self.TE = self.tdiff + self.delta  # Simple approximation
        
        # Handle gradient directions
        if gdir is not None:
            self.gdir = np.asarray(gdir)
            assert self.gdir.shape == (n_meas, 3), f"gdir shape {self.gdir.shape} != ({n_meas}, 3)"
        else:
            # Default: all along z-axis except b=0
            self.gdir = np.zeros((n_meas, 3))
            self.gdir[:, 2] = 1.0
            self.gdir[self.b == 0] = 0.0
        
        # Handle gradient strengths
        if G is not None:
            if np.isscalar(G):
                self.G = np.full(n_meas, G)
            else:
                self.G = np.asarray(G)
        else:
            # Calculate from b-values if not provided
            self.G = np.zeros(n_meas)
            for i in range(n_meas):
                if self.b[i] > 0:
                    # From b = (γGδ)² (Δ - δ/3)
                    b_SI = self.b[i] * 1e6  # Convert to s/m²
                    self.G[i] = np.sqrt(b_SI / ((self.tdiff[i] - self.delta[i]/3) * 
                                                 (GYRO * self.delta[i])**2))
    
    def get_unique_timings(self):
        """Get unique (delta, tdiff) pairs"""
        timings = np.column_stack((self.delta, self.tdiff))
        return np.unique(timings, axis=0)
    
    def normalize_sig(self, sig):
        """Normalize signal to b=0 measurements"""
        b0_indices = np.where(self.b == 0)[0]
        if len(b0_indices) > 0:
            s0 = np.mean(sig[b0_indices])
            return sig / s0
        return sig

# ============================================================================
# SPHERE SIGNAL MODEL - CANONICAL SERIES IMPLEMENTATION
# ============================================================================

def sphere_signal_series(R, D, b, delta, tdiff, n_max=32):
    """
    Calculate restricted diffusion signal in spheres using the GPD approximation
    (van Gelderen / Murday-Cotts) for a PGSE sequence.

    Uses the reflecting-boundary eigenvalues (zeros of j'_1).
    
    Parameters:
    -----------
    R : float
        Sphere radius in meters
    D : float
        Free diffusivity in m²/s
    b : float
        b-value in s/mm²
    delta : float
        Gradient duration in seconds
    tdiff : float
        Diffusion time (Δ) in seconds
    n_max : int
        Number of terms in series
        
    Returns:
    --------
    signal : float
        Normalized signal (S/S0)
    """
    
    # Lambda values from Bessel function roots
    lambda_k = BES_ROOTS[:n_max] / R  # 1/m
    am = lambda_k**2  # 1/m²
    
    # GPD denominator (van Gelderen):  D^2 * alpha_k^6 * (alpha_k^2 R^2 - 2)
    denom = D**2 * am**3 * (am * R**2 - 2)
    
    # Time-dependent terms - include D in the exponential arguments
    sda2 = D * delta * am  # D * δ * α² (dimensionless)
    bda2 = D * tdiff * am  # D * Δ * α² (dimensionless)
    
    # Exponential terms with overflow protection
    emdsda2 = np.exp(np.maximum(-sda2, -50))
    emdbda2 = np.exp(np.maximum(-bda2, -50))
    emdbdmsda2 = np.exp(np.maximum(-(bda2 - sda2), -50))
    emdbdpsda2 = np.exp(np.maximum(-(bda2 + sda2), -50))
    
    # Numerator (GPD formula)
    num = 2*sda2 - 2 + 2*emdsda2 + 2*emdbda2 - emdbdmsda2 - emdbdpsda2
    
    # Avoid division by zero
    valid = np.abs(denom) > 1e-30
    
    # Sum over modes
    sumt = np.sum(num[valid] / denom[valid])
    
    # Convert to signal using correct GPD formula
    # For PGSE: b = γ²G²δ²(Δ - δ/3)
    b_SI = b * 1e6  # Convert s/mm² to s/m²
    
    # Calculate γ²G² from b-value and timing parameters
    gamma2_G2 = b_SI / (delta**2 * (tdiff - delta/3))
    
    # GPD signal: E = exp(-2 γ²G² ∑(num/denom))
    logE = -2 * gamma2_G2 * sumt
    signal = np.exp(logE)
    
    # Ensure signal is in valid range
    signal = np.clip(signal, 1e-10, 1.0)
    
    return signal

def SPHERE_sig(R, D, protocol, normalize=False):
    """
    Restricted diffusion signal in spheres - full protocol.
    
    Parameters:
    -----------
    R : float
        Sphere radius in μm
    D : float
        Diffusivity in m²/s
    protocol : Protocol
        Acquisition protocol
    normalize : bool
        Normalize to b=0
        
    Returns:
    --------
    S : array
        Signal for each measurement
    """
    
    # Validate inputs
    if R <= 0 or R > 20:
        warnings.warn(f"Sphere radius R={R} μm outside typical range [0.1, 20] μm")
    if D <= 0 or D > 3e-9:
        warnings.warn(f"Diffusivity D={D:.2e} m²/s outside typical range [1e-12, 3e-9] m²/s")
    
    # Initialize signal
    S = np.ones(len(protocol.b), dtype=np.float64)
    
    # Convert radius to meters
    R_m = R * MUM_TO_M
    
    # Process each measurement
    for i in range(len(protocol.b)):
        if protocol.b[i] == 0:
            S[i] = 1.0
        else:
            S[i] = sphere_signal_series(
                R_m, D, 
                protocol.b[i], 
                protocol.delta[i], 
                protocol.tdiff[i]
            )
    
    if normalize:
        S = protocol.normalize_sig(S)
    
    return S

# ============================================================================
# CEXI MODEL - IMPLEMENTATION
# ============================================================================

def CEXI_sig(R, Di, De, f, kappa, protocol, normalize=False):
    """
    CEXI signal model: restricted-sphere intracellular + Gaussian extracellular
    compartments coupled by Kärger water exchange.

    Parameters:
    -----------
    R : float
        Sphere radius (μm)
    Di : float
        Intracellular diffusivity (m²/s)
    De : float
        Extracellular diffusivity (m²/s)
    f : float
        Intracellular volume fraction (0-1)
    kappa : float
        Membrane permeability (μm/s)
    protocol : Protocol
        Acquisition protocol with per-measurement parameters
    normalize : bool
        Normalize to b=0
        
    Returns:
    --------
    S : array
        Signal for each measurement
    """
    
    # Validate parameters
    if R <= 0 or R > 20:
        warnings.warn(f"Sphere radius R={R} μm outside typical range [0.1, 20] μm")
    if Di <= 0 or Di > 3e-9:
        warnings.warn(f"Intracellular diffusivity Di={Di:.2e} m²/s outside typical range")
    if De <= 0 or De > 3e-9:
        warnings.warn(f"Extracellular diffusivity De={De:.2e} m²/s outside typical range")
    if f <= 0 or f >= 1:
        warnings.warn(f"Volume fraction f={f} should be in (0, 1)")
    if kappa > PERM_MAX_STABLE:
        warnings.warn(f"Permeability κ={kappa} μm/s above stable range. "
                     f"Paper reports instability above {PERM_MAX_STABLE} μm/s")
    
    # Initialize signal
    S = np.ones(len(protocol.b), dtype=np.float64)
    
    # Process each measurement
    for i in range(len(protocol.b)):
        if protocol.b[i] == 0:
            S[i] = 1.0
            continue
            
        # Get parameters for this measurement
        b = protocol.b[i]
        delta = protocol.delta[i]
        tdiff = protocol.tdiff[i]
        
        # Calculate q² for exchange model
        q2 = b * 1e6 / (tdiff - delta/3)  # Convert to SI units (1/m²)
        
        # Intracellular signal with restriction
        Si_restricted = sphere_signal_series(
            R * MUM_TO_M, Di, b, delta, tdiff
        )
        
        # Calculate apparent Di
        Di_app = -np.log(np.maximum(Si_restricted, EPS)) / (b * 1e6)  # m²/s
        
        # Convert units for exchange calculation
        kappa_SI = kappa * MUM_TO_M  # μm/s to m/s
        R_SI = R * MUM_TO_M          # μm to m
        
        # Check if permeable
        if kappa_SI < PERM_MIN * MUM_TO_M:
            # Impermeable case - no exchange
            S[i] = f * np.exp(-Di_app * b * 1e6) + (1-f) * np.exp(-De * b * 1e6)
        else:
            # Kärger exchange rates.
            # Intracellular efflux rate is a LOCAL cell property and depends only
            # on permeability and surface-to-volume ratio (3/R for a sphere):
            #     k_i = kappa * (S/V) = 3 kappa / R        (independent of f)
            # The influx rate follows from detailed balance (mass conservation):
            #     f * k_i = (1 - f) * k_e   =>   k_e = k_i * f / (1 - f)
            ki = 3.0 * kappa_SI / R_SI              # 1/s, intracellular efflux
            ke = ki * f / (1.0 - f)                 # 1/s, extracellular influx

            # Exchange times
            tie = 1.0 / ki  # Intracellular residence time  R/(3 kappa)
            tei = 1.0 / ke  # Extracellular residence time
            
            q2tie = q2 * tie
            q2tei = q2 * tei
            
            # Kärger model for exchange
            xe = De + 1/q2tei
            xi = Di_app + 1/q2tie
            c = xi + xe
            r = np.sqrt((xi - xe)**2 + 4/(q2tie * q2tei))
            
            # Apparent diffusivities with exchange
            Di_ex = 0.5 * (c - r)
            De_ex = 0.5 * (c + r)
            
            # Apparent volume fractions
            fe = (f * Di_app + (1-f) * De - Di_ex) / r
            fi = 1 - fe
            
            S[i] = fi * np.exp(-Di_ex * b * 1e6) + fe * np.exp(-De_ex * b * 1e6)
    
    if normalize:
        S = protocol.normalize_sig(S)
    
    return S

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_noisy_signal(clean_signal, SNR=20, noise_type='rician'):
    """
    Add noise to clean signal
    
    Parameters:
    -----------
    clean_signal : array
        Clean signal values
    SNR : float
        Signal-to-noise ratio (at b=0)
    noise_type : str
        'gaussian' or 'rician'
        
    Returns:
    --------
    noisy_signal : array
        Signal with added noise
    """
    
    # Calculate noise standard deviation
    signal_b0 = clean_signal[0] if len(clean_signal) > 0 else 1.0
    noise_std = signal_b0 / SNR
    
    if noise_type == 'gaussian':
        # Simple Gaussian noise
        noise = np.random.normal(0, noise_std, clean_signal.shape)
        noisy_signal = clean_signal + noise
        
    elif noise_type == 'rician':
        # Rician noise (more realistic for MRI)
        noise_real = np.random.normal(0, noise_std, clean_signal.shape)
        noise_imag = np.random.normal(0, noise_std, clean_signal.shape)
        noisy_signal = np.sqrt((clean_signal + noise_real)**2 + noise_imag**2)
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy_signal

# ============================================================================
# MODEL FITTING
# ============================================================================

# Parameter order used by the fitter and the encode/decode transforms.
FIT_PARAM_KEYS = ['R', 'Di', 'De', 'f', 'kappa']

# Parameters optimized on a logarithmic axis. The CEXI signal spans several
# orders of magnitude in permeability and the diffusivities differ from the
# radius by ~9 orders of magnitude (m²/s vs μm); normalizing every parameter
# to [0, 1] - and using a log axis where sensitivity is ~log - makes the
# finite-difference gradients well conditioned for L-BFGS-B.
FIT_LOG_KEYS = ('Di', 'De', 'kappa')


def _make_param_transforms(bounds, log_keys=FIT_LOG_KEYS):
    """
    Build per-parameter encode (physical -> [0, 1]) and decode ([0, 1] ->
    physical) functions from a bounds dict. Linear within bounds, except for
    `log_keys` which are mapped on a logarithmic axis.

    Returns
    -------
    encode : callable(dict-or-list) -> np.ndarray   physical -> unit cube
    decode : callable(np.ndarray)   -> dict         unit cube -> physical
    """
    los, his, is_log = [], [], []
    for k in FIT_PARAM_KEYS:
        lo, hi = bounds[k]
        use_log = (k in log_keys) and (lo > 0)
        if use_log:
            lo, hi = np.log(lo), np.log(hi)
        los.append(lo); his.append(hi); is_log.append(use_log)
    los, his, is_log = np.array(los), np.array(his), np.array(is_log)
    span = np.where(his > los, his - los, 1.0)

    def encode(params):
        x = np.array([params[k] for k in FIT_PARAM_KEYS], dtype=float) \
            if isinstance(params, dict) else np.asarray(params, dtype=float)
        x = np.where(is_log, np.log(np.clip(x, EPS, None)), x)
        return np.clip((x - los) / span, 0.0, 1.0)

    def decode(u):
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        x = los + u * span
        x = np.where(is_log, np.exp(x), x)
        return {k: float(v) for k, v in zip(FIT_PARAM_KEYS, x)}

    return encode, decode


def fit_CEXI(signal_data, protocol, bounds=None, n_init=12, x0=None, seed=None):
    """
    Fit the CEXI model to signal data.

    The optimization is carried out in a normalized [0, 1] coordinate per
    parameter (linear for R and f, logarithmic for Di, De and kappa). This
    removes the ~9-order-of-magnitude scale gap between the radius (μm) and the
    diffusivities (m²/s) that otherwise makes the raw-parameter L-BFGS-B
    gradients ill-conditioned and stalls the fit.

    Parameters
    ----------
    signal_data : array
        Measured signal values.
    protocol : Protocol
        Acquisition protocol.
    bounds : dict, optional
        Parameter bounds with keys 'R', 'Di', 'De', 'f', 'kappa'.
    n_init : int
        Number of random restarts (multi-start).
    x0 : dict, optional
        Optional explicit first guess (physical units); used in addition to
        the random restarts.
    seed : int, optional
        Seed for the restart RNG, for reproducible fits.

    Returns
    -------
    fitted_params : dict or None
        Best-fit parameters in physical units.
    fit_cost : float
        Final cost (RMSE on the normalized signal).
    """

    # Default parameter bounds
    if bounds is None:
        bounds = {
            'R': (2.0, 15.0),       # μm
            'Di': (0.3e-9, 2.0e-9), # m²/s
            'De': (1.0e-9, 3.5e-9), # m²/s
            'f': (0.3, 0.85),       # fraction
            'kappa': (1.0, 100.0)   # μm/s
        }

    encode, decode = _make_param_transforms(bounds)

    # Normalize signal once
    signal_norm = protocol.normalize_sig(signal_data)

    # Objective in normalized coordinates: u in [0, 1]^5
    def objective(u):
        p = decode(u)
        try:
            signal_model = CEXI_sig(p['R'], p['Di'], p['De'], p['f'],
                                    p['kappa'], protocol, normalize=True)
            res = np.sqrt(np.mean((signal_norm - signal_model) ** 2))
            return res if np.isfinite(res) else 1e10
        except Exception:
            return 1e10

    rng = np.random.default_rng(seed)
    cube_bounds = [(0.0, 1.0)] * len(FIT_PARAM_KEYS)

    # Build the list of starting points (encoded): optional explicit x0 first,
    # then random restarts drawn uniformly in the unit cube.
    starts = []
    if x0 is not None:
        starts.append(encode(x0))
    starts.extend(rng.random(len(FIT_PARAM_KEYS)) for _ in range(n_init))

    best_u, best_cost = None, np.inf
    for u0 in starts:
        result = minimize(objective, u0, method='L-BFGS-B', bounds=cube_bounds)
        if result.fun < best_cost:
            best_cost = result.fun
            best_u = result.x

    if best_u is not None:
        return decode(best_u), float(best_cost)

    return None, np.inf

# ============================================================================
# SCHEME FILE I/O (unchanged)
# ============================================================================

def load_scheme_file(scheme_path):
    """Load MCDC-style scheme file"""
    
    gdir_list = []
    G_list = []
    delta_list = []
    tdiff_list = []
    
    with open(scheme_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('VERSION') or not line:
            continue
            
        parts = line.split()
        if len(parts) >= 6:
            gx, gy, gz = float(parts[0]), float(parts[1]), float(parts[2])
            G = float(parts[3])
            Delta = float(parts[4])
            delta = float(parts[5])
            
            gdir_list.append([gx, gy, gz])
            G_list.append(G)
            delta_list.append(delta)
            tdiff_list.append(Delta)
    
    gdir = np.array(gdir_list)
    G = np.array(G_list)  
    delta = np.array(delta_list)
    tdiff = np.array(tdiff_list)
    
    b_values = []
    for i in range(len(G)):
        if G[i] == 0:
            b_val = 0
        else:
            b_SI = (GYRO * G[i] * delta[i])**2 * (tdiff[i] - delta[i]/3)
            b_val = b_SI * 1e-6
        b_values.append(b_val)
    
    b_values = np.array(b_values)
    
    protocol = Protocol(
        bvals=b_values,
        delta=delta,
        tdiff=tdiff,
        gdir=gdir,
        G=G
    )
    
    return protocol