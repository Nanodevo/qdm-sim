"""The diamond sensor: how growth and treatment parameters become NV density, coherence,
strain and contrast, i.e. the numbers the rest of the model consumes.

Simple, published scalings, each with its source in the docstring. The point is the
direction and size of each lever, not a process model of a particular reactor.
"""
from __future__ import annotations

import numpy as np

PPM_TO_CM3 = 1.76e17          # one ppm of substitutional nitrogen in diamond, atoms per cm^3
A_N_HZ_PER_PPM = 2 * np.pi * 14e3   # dephasing rate per ppm of P1 nitrogen (Bauch et al. 2020, N-bath limit)
A_C13_HZ = 2 * np.pi * 100e3        # dephasing rate from the 13C bath at natural abundance (1.1 %)
C13_NATURAL = 0.011
RESIDUAL_HZ = 2 * np.pi * 15e3        # residual dephasing from strain, electric fields and other defects: T2* ceiling ~10 us
C_MAX = 0.03                        # ensemble CW contrast with the whole layer in the NV- state


def nv_density_cm3(n_ppm: float, dose_1e18_cm2: float, yield_max: float = 0.10, dose0_1e18: float = 1.0) -> float:
    """NV- density from nitrogen content, electron-irradiation dose and annealing.

    Vacancies are created in proportion to the dose and captured by nitrogen on annealing
    (800-1000 C); the N -> NV conversion yield saturates because the nitrogen supply and the
    vacancy capture both run out: y(dose) = yield_max (1 - exp(-dose/dose0)). Typical yields
    are a few percent for CVD layers and up to ~10 % with optimised irradiation."""
    y = yield_max * (1 - np.exp(-dose_1e18_cm2 / dose0_1e18))
    return float(n_ppm * PPM_TO_CM3 * y)


def t2_star_s(n_ppm: float, c13_fraction: float = C13_NATURAL, strain_broadening_hz: float = 0.0) -> float:
    """Ensemble dephasing time from the nitrogen bath, the 13C bath, a residual term (strain,
    electric fields, other defects) and any extra inhomogeneous broadening:
    1/T2* = A_N [N] + A_C (x13 / 1.1 %) + residual + pi * delta_strain."""
    rate = A_N_HZ_PER_PPM * n_ppm + A_C13_HZ * (c13_fraction / C13_NATURAL) + RESIDUAL_HZ + np.pi * strain_broadening_hz
    return float(1.0 / rate)


def linewidth_hz(t2_star: float) -> float:
    """Intrinsic ODMR linewidth (FWHM) of a Lorentzian line with dephasing time T2*: 1/(pi T2*)."""
    return float(1.0 / (np.pi * t2_star))


def active_layer_um(layer_um: float, surface: str = "oxygen") -> float:
    """NV- survives only beyond a surface dead layer set by the termination and its band
    bending: ~5 nm for oxygen-terminated, ~30 nm and more for hydrogen-terminated surfaces."""
    dead_nm = {"oxygen": 5.0, "hydrogen": 30.0, "fluorine": 3.0}[surface]
    return float(max(layer_um - dead_nm * 1e-3, 0.0))


def contrast(nv_minus_fraction: float = 0.7, readout_efficiency: float = 1.0) -> float:
    """CW ODMR contrast scales with the NV- charge fraction (NV0 fluoresces but carries no
    spin signal) and with how well the readout window catches the spin-dependent part."""
    return float(C_MAX * nv_minus_fraction / 0.7 * readout_efficiency)


def strain_splitting_hz(strain_ppm: float) -> float:
    """Zero-field splitting E from crystal strain, order of magnitude 1 MHz per ppm of lattice
    strain; shows as a line splitting (single NV) or broadening (ensemble)."""
    return float(1e6 * strain_ppm)
