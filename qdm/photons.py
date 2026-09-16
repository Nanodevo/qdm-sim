"""From illumination to collected photons to shot-noise-limited sensitivity."""
from __future__ import annotations

import numpy as np

N_DIAMOND = 2.417
R_SAT_PER_NV = 1.0e7        # emitted photons per second per NV at saturation (radiative lifetime ~12 ns, branching to the phonon sideband included)
I_SAT_W_CM2 = 1.0e6         # saturation intensity of the 532 nm excitation, single-NV order of magnitude
FRESNEL_T_NORMAL = 1 - ((N_DIAMOND - 1) / (N_DIAMOND + 1)) ** 2   # 0.83


def escape_cone_fraction(n: float = N_DIAMOND) -> float:
    """Fraction of an isotropic emitter's light inside the critical-angle cone of one surface."""
    theta_c = np.arcsin(1.0 / n)
    return float((1 - np.cos(theta_c)) / 2)


def collection_fraction(na: float, n: float = N_DIAMOND, sil: bool = False, back_mirror: bool = False) -> float:
    """Fraction of emitted photons an objective of numerical aperture `na` collects from an
    NV layer just below a planar diamond surface (isotropic emission assumed).

    Bare surface: the acceptance half-angle inside the diamond is asin(na/n), and the
    Fresnel transmission at the surface is applied. Solid immersion lens (index matched,
    anti-reflection coated hemisphere): the half-angle is asin(na) with no refraction loss.
    A reflective back side folds the downward half into the upward one (ideal mirror)."""
    if sil:
        theta = np.arcsin(min(na, 1.0))
        t = 1.0
    else:
        theta = np.arcsin(min(na / n, 1.0))
        t = FRESNEL_T_NORMAL
    frac = (1 - np.cos(theta)) / 2 * t
    return float(2 * frac if back_mirror else frac)


def emission_rate_per_nv(intensity_w_cm2: float) -> float:
    return R_SAT_PER_NV * intensity_w_cm2 / (intensity_w_cm2 + I_SAT_W_CM2)


def nv_per_pixel(density_cm3: float, layer_um: float, pixel_um: float) -> float:
    return density_cm3 * (layer_um * 1e-4) * (pixel_um * 1e-4) ** 2


def photon_rate_per_pixel(intensity_w_cm2, density_cm3, layer_um, pixel_um, collection, optics_t=0.7, qe=0.8) -> float:
    """Detected photons per second in one camera pixel (object-side pixel size)."""
    return nv_per_pixel(density_cm3, layer_um, pixel_um) * emission_rate_per_nv(intensity_w_cm2) * collection * optics_t * qe


def sensitivity_t_per_rthz(rate_per_s, contrast=0.02, linewidth_hz=1.0e6, gamma_hz_per_t=28.024e9) -> float:
    """Shot-noise-limited CW-ODMR field sensitivity, tesla per root hertz (Dreau et al. 2011):
    eta = (4/(3 sqrt 3)) * linewidth / (gamma * contrast * sqrt(rate))."""
    return float(4 / (3 * np.sqrt(3)) * linewidth_hz / (gamma_hz_per_t * contrast * np.sqrt(rate_per_s)))


def field_noise(eta_t_per_rthz, t_meas_s) -> float:
    """Standard deviation of one field value after `t_meas_s` seconds of averaging."""
    return float(eta_t_per_rthz / np.sqrt(t_meas_s))
