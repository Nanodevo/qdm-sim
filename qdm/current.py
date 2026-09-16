"""Currents in a chip -> field at the NV plane -> current map back from the field.

Sheet currents J (A/m) in a plane a distance d below the NV layer. Forward field by the
Fourier form of Biot-Savart for the out-of-plane component (Roth, Sepulveda and Wikswo
1989):

    Bz(k) = (mu0/2) exp(-k d) (i kx Jy(k) - i ky Jx(k)) / k

A divergence-free sheet current derives from a stream function g: Jx = dg/dy,
Jy = -dg/dx, which turns the forward operator into Bz(k) = (mu0/2) k exp(-k d) g(k), and
the inversion into g(k) = 2 Bz(k) exp(k d) / (mu0 k) with a low-pass window, because the
exp(k d) amplifies every spatial frequency the stand-off has attenuated, noise included.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


def grid(n: int, dx_um: float):
    """Centred coordinate axes in micrometres."""
    x = (np.arange(n) - n // 2) * dx_um
    return x, x.copy()


def _kgrid(n: int, dx_m: float):
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=dx_m)
    kx, ky = np.meshgrid(k1, k1, indexing="xy")
    return kx, ky, np.sqrt(kx ** 2 + ky ** 2)


def rounded_rect_stream(x_um, y_um, cx, cy, w, h, current_a, edge_um):
    """Stream function of a current loop: `current_a` circulates around a w x h rectangle
    centred at (cx, cy); the wire has a smooth profile of width ~edge_um."""
    X, Y = np.meshgrid(x_um, y_um, indexing="xy")
    fx = 0.5 * (np.tanh((X - (cx - w / 2)) / edge_um) - np.tanh((X - (cx + w / 2)) / edge_um))
    fy = 0.5 * (np.tanh((Y - (cy - h / 2)) / edge_um) - np.tanh((Y - (cy + h / 2)) / edge_um))
    return current_a * fx * fy


def currents_from_stream(g_a, dx_um):
    """Jx = dg/dy, Jy = -dg/dx, in A/m, from a stream function in amperes."""
    dx = dx_um * 1e-6
    gy, gx = np.gradient(g_a, dx, dx)   # gradient returns d/d(axis0)=y first, then x
    return gy, -gx


def bz_from_sheet(jx, jy, dx_um, standoff_um):
    """Out-of-plane field (tesla) at height d above the sheet."""
    n = jx.shape[0]
    kx, ky, k = _kgrid(n, dx_um * 1e-6)
    d = standoff_um * 1e-6
    Jx, Jy = np.fft.fft2(jx), np.fft.fft2(jy)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bz = (MU0 / 2) * np.exp(-k * d) * (1j * kx * Jy - 1j * ky * Jx) / k
    Bz[k == 0] = 0.0
    return np.real(np.fft.ifft2(Bz))


def bz_biot_savart(jx, jy, dx_um, standoff_um):
    """Direct convolution with the Biot-Savart kernel (slow, for checking the Fourier form)."""
    n = jx.shape[0]
    dx = dx_um * 1e-6
    d = standoff_um * 1e-6
    ax = (np.arange(2 * n - 1) - (n - 1)) * dx
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    r3 = (X ** 2 + Y ** 2 + d ** 2) ** 1.5
    gx, gy = X / r3, Y / r3
    from scipy.signal import fftconvolve
    return MU0 / (4 * np.pi) * (fftconvolve(jx, gy, mode="same") - fftconvolve(jy, gx, mode="same")) * dx * dx


def hanning_window(k, k_cut):
    w = 0.5 * (1 + np.cos(np.pi * k / k_cut))
    w[k > k_cut] = 0.0
    return w


def reconstruct_currents(bz_t, dx_um, standoff_um, k_cut_per_um: float | None = None):
    """Sheet currents (A/m) from a Bz map (tesla) at stand-off d, with a Hanning low-pass at
    k_cut (default 1/d rad per um, ~ the point where noise amplification exp(k d) reaches e)."""
    n = bz_t.shape[0]
    dx = dx_um * 1e-6
    d = standoff_um * 1e-6
    kx, ky, k = _kgrid(n, dx)
    k_cut = (k_cut_per_um if k_cut_per_um else 1.0 / standoff_um) * 1e6
    B = np.fft.fft2(bz_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = 2 * B * np.exp(k * d) / (MU0 * k) * hanning_window(k, k_cut)
    G[k == 0] = 0.0
    Jx = np.real(np.fft.ifft2(1j * ky * G))
    Jy = np.real(np.fft.ifft2(-1j * kx * G))
    return Jx, Jy


def rms_error(j_rec, j_true):
    m = np.hypot(*j_true)
    r = np.hypot(*j_rec)
    return float(np.sqrt(np.mean((r - m) ** 2)) / np.sqrt(np.mean(m ** 2)))
