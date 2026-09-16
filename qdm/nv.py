"""Nitrogen-vacancy ground-state spin physics: levels, ODMR lines, spectra.

H/h = D Sz^2 + gamma (B . S) for the spin-1 ground state, one Hamiltonian per NV
orientation. Four orientations along the <111> directions of the diamond lattice.
Constants: D = 2.870 GHz zero-field splitting, gamma = 28.024 GHz/T.
"""
from __future__ import annotations

import numpy as np

D_GHZ = 2.870
GAMMA_GHZ_PER_T = 28.024
HYPERFINE_14N_MHZ = 2.16

_s = 1 / np.sqrt(2)
SX = np.array([[0, _s, 0], [_s, 0, _s], [0, _s, 0]], dtype=complex)
SY = np.array([[0, -1j * _s, 0], [1j * _s, 0, -1j * _s], [0, 1j * _s, 0]], dtype=complex)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)

# the four NV axes in the crystal frame (unit vectors)
NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)
# for a (100)-cut plate the surface normal is [001]; every NV axis makes 54.7 deg with it


def hamiltonian_ghz(b_T: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Spin-1 Hamiltonian in GHz in the frame of one NV axis (z along the axis)."""
    b = np.asarray(b_T, float)
    b_par = float(np.dot(b, axis))
    b_perp = float(np.linalg.norm(b - b_par * axis))
    return D_GHZ * SZ @ SZ + GAMMA_GHZ_PER_T * (b_par * SZ + b_perp * SX)


def transitions_ghz(b_T, axis) -> tuple[float, float]:
    """The two ms=0 -> ms=+-1 transition frequencies of one orientation, in GHz."""
    H = hamiltonian_ghz(b_T, axis)
    e, v = np.linalg.eigh(H)
    # the ms=0-like state is the eigenvector with the largest weight on |0> (index 1)
    i0 = int(np.argmax(np.abs(v[1, :]) ** 2))
    others = [i for i in range(3) if i != i0]
    f = sorted(float(e[i] - e[i0]) for i in others)
    return f[0], f[1]


def all_transitions_ghz(b_T) -> np.ndarray:
    """Eight lines, ordered by NV orientation then frequency."""
    return np.array([transitions_ghz(b_T, ax) for ax in NV_AXES])


def lorentzian(f, f0, fwhm):
    return 1.0 / (1.0 + (2 * (f - f0) / fwhm) ** 2)


def odmr_spectrum(f_ghz, b_T, contrast=0.02, linewidth_mhz=1.0, hyperfine=False) -> np.ndarray:
    """Normalised fluorescence versus microwave frequency for an ensemble with all four
    orientations equally populated. `contrast` is the dip depth of a fully resolved
    single line if all NVs were of one orientation; each of the eight lines therefore
    carries contrast/4 (and each 14N hyperfine sub-line a third of that)."""
    f = np.asarray(f_ghz, float)
    w = linewidth_mhz * 1e-3
    sig = np.ones_like(f)
    for lines in all_transitions_ghz(b_T):
        for f0 in lines:
            if hyperfine:
                for df in (-HYPERFINE_14N_MHZ, 0.0, HYPERFINE_14N_MHZ):
                    sig -= (contrast / 4 / 3) * lorentzian(f, f0 + df * 1e-3, w)
            else:
                sig -= (contrast / 4) * lorentzian(f, f0, w)
    return sig


def field_from_axis_splitting(f_plus_ghz: float, f_minus_ghz: float) -> float:
    """Field component along the NV axis, in tesla, from the two line positions (small
    transverse field): f+ - f- = 2 gamma B_par."""
    return (f_plus_ghz - f_minus_ghz) / (2 * GAMMA_GHZ_PER_T)
