"""Imaging resolution and the stand-off limit."""
from __future__ import annotations

import numpy as np

LAMBDA_EMISSION_UM = 0.70        # NV phonon-sideband emission, centre of the collected band
MU0 = 4e-7 * np.pi


def diffraction_fwhm_um(na: float, wavelength_um: float = LAMBDA_EMISSION_UM) -> float:
    """Airy-disc FWHM of the imaging system in the object plane, ~0.51 lambda / NA."""
    return 0.51 * wavelength_um / na


def psf_sigma_um(na: float, wavelength_um: float = LAMBDA_EMISSION_UM) -> float:
    """Gaussian approximation of the point-spread function, sigma ~ 0.21 lambda / NA."""
    return 0.21 * wavelength_um / na


def wire_bz(x_um, current_a: float, standoff_um: float) -> np.ndarray:
    """Out-of-plane field of an infinite straight wire along y at depth d below the NV plane:
    Bz(x) = mu0 I / (2 pi) * x / (x^2 + d^2). Peaks sit at x = +-d, so the peak-to-peak
    separation is 2d whatever the optics do."""
    x = np.asarray(x_um, float) * 1e-6
    d = standoff_um * 1e-6
    return MU0 * current_a / (2 * np.pi) * x / (x ** 2 + d ** 2)


def wire_response_width_um(na: float, standoff_um: float, wavelength_um: float = LAMBDA_EMISSION_UM) -> float:
    """Peak-to-peak separation of the imaged wire response: the stand-off's 2d combined in
    quadrature with the optical resolution."""
    return float(np.sqrt((2 * standoff_um) ** 2 + diffraction_fwhm_um(na, wavelength_um) ** 2))


def standoff_transfer(k_per_um, standoff_um) -> np.ndarray:
    """Spatial-frequency transfer of the stand-off: exp(-k d). Half the amplitude is gone at
    k = ln 2 / d, i.e. at a period of 2 pi d / ln 2 ~ 9 d."""
    return np.exp(-np.asarray(k_per_um, float) * standoff_um)
