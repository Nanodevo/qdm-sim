"""Measurement protocols and their shot-noise-limited sensitivities.

All three reduce to eta = (phase or frequency resolution) / (gamma * contrast * sqrt(photons)),
they differ in what sets the resolution and how many photons the duty cycle leaves.
CW ODMR: the line, power-broadened by the microwave drive (Dreau et al. 2011).
Pulsed ODMR: the line at its T2*-limited width, paid for with a readout duty cycle.
Ramsey: a free-precession time tau ~ T2* instead of a linewidth (Barry et al. 2020).
"""
from __future__ import annotations

import numpy as np

GAMMA_HZ_PER_T = 28.024e9
K_CW = 4 / (3 * np.sqrt(3))


def cw_line(s: float, linewidth0_hz: float, contrast0: float) -> tuple[float, float]:
    """Power-broadened width and contrast at saturation parameter s = (Omega_R / Omega_sat)^2."""
    return float(linewidth0_hz * np.sqrt(1 + s)), float(contrast0 * s / (1 + s))


def eta_cw(rate_per_s: float, linewidth0_hz: float, contrast0: float, s: float = 2.0) -> float:
    """CW ODMR sensitivity, tesla per root hertz, at microwave saturation s (optimum s = 2)."""
    w, c = cw_line(s, linewidth0_hz, contrast0)
    return float(K_CW * w / (GAMMA_HZ_PER_T * c * np.sqrt(rate_per_s)))


def eta_pulsed_odmr(rate_per_s: float, t2_star_s: float, contrast0: float, t_read_s: float = 1e-6, t_init_s: float = 2e-6) -> float:
    """Pulsed ODMR: a pi pulse of length ~T2* probes the unbroadened line 1/(pi T2*); photons
    are only collected during the readout window, so the rate is scaled by the duty cycle."""
    t_pi = t2_star_s
    t_cycle = t_init_s + t_pi + t_read_s
    duty = t_read_s / t_cycle
    w = 1.0 / (np.pi * t2_star_s)
    return float(K_CW * w / (GAMMA_HZ_PER_T * contrast0 * np.sqrt(rate_per_s * duty)))


def eta_ramsey(rate_per_s: float, t2_star_s: float, contrast0: float, t_read_s: float = 1e-6, t_init_s: float = 2e-6, tau_over_t2: float = 0.5) -> float:
    """Ramsey: phase 2 pi gamma B tau accumulated over tau, read out with contrast
    C exp(-tau/T2*). Per cycle dB = dphi / (2 pi gamma tau) with dphi = 1/(C_eff sqrt(N_ph));
    cycles repeat every t_cycle, so eta = dB_cycle * sqrt(t_cycle)."""
    tau = tau_over_t2 * t2_star_s
    c_eff = contrast0 * np.exp(-tau / t2_star_s)
    n_ph = rate_per_s * t_read_s
    t_cycle = t_init_s + tau + t_read_s
    db_cycle = 1.0 / (c_eff * np.sqrt(n_ph)) / (2 * np.pi * GAMMA_HZ_PER_T * tau)
    return float(db_cycle * np.sqrt(t_cycle))
