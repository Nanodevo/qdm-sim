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


# ---------------------------------------------------------------- two layers at known depths
def stream_from_bz(bz_t, dx_um, d_um, k_cut_per_um):
    """Single-layer windowed inversion, returned as the stream function (amperes)."""
    n = bz_t.shape[0]
    dx = dx_um * 1e-6
    d = d_um * 1e-6
    kx, ky, k = _kgrid(n, dx)
    B = np.fft.fft2(bz_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = 2 * B * np.exp(k * d) / (MU0 * k) * hanning_window(k, k_cut_per_um * 1e6)
    G[k == 0] = 0.0
    return np.real(np.fft.ifft2(G))


def _forward_stream(g_a, dx_um, d_um):
    jx, jy = currents_from_stream(g_a, dx_um)
    return bz_from_sheet(jx, jy, dx_um, d_um)


def _tikhonov_pair(bz_t, dx_um, d1_um, d2_um, lam1, lam2):
    """Per spatial frequency, the least-squares split of Bz between two current sheets at known
    depths, each with its own gradient penalty: minimise |B - a1 g1 - a2 g2|^2 + k^2 (lam1 |g1|^2 + lam2 |g2|^2),
    a_i = (mu0/2) k exp(-k d_i). High spatial frequencies can only come from the shallow layer;
    low ones are shared according to the penalties, which is the residual ambiguity."""
    n = bz_t.shape[0]
    kx, ky, k = _kgrid(n, dx_um * 1e-6)
    B = np.fft.fft2(bz_t)
    a1 = (MU0 / 2) * k * np.exp(-k * d1_um * 1e-6)
    a2 = (MU0 / 2) * k * np.exp(-k * d2_um * 1e-6)
    r1, r2 = lam1 * k ** 2, lam2 * k ** 2
    det = (a1 ** 2 + r1) * (a2 ** 2 + r2) - (a1 * a2) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = ((a2 ** 2 + r2) * a1 * B - a1 * a2 * a2 * B) / det
        g2 = ((a1 ** 2 + r1) * a2 * B - a1 * a2 * a1 * B) / det
    g1[k == 0] = 0.0; g2[k == 0] = 0.0
    return np.real(np.fft.ifft2(g1)), np.real(np.fft.ifft2(g2))


def reconstruct_two_layers(bz_t, dx_um, d1_um, d2_um, snr: float, masks=None, n_iter: int = 30):
    """Two current layers at known depths from one Bz map.

    Without masks: a spectral least-squares split with a per-layer penalty tuned so that each
    layer's cut-off sits where the depth amplification of the noise reaches the signal. With
    masks (one boolean array per layer marking where that metal level has wires, i.e. the layout):
    alternating single-layer inversions, each layer solved from the residual left by the other
    and clipped to its allowed region. Because the two levels do not occupy the same places, the
    low-frequency ambiguity of the spectral split disappears."""
    lsnr = max(np.log(max(snr, 1.5)), 1.0)
    kc1, kc2 = lsnr / d1_um, lsnr / d2_um
    if masks is not None:
        kc1, kc2 = 2 * kc1, 2 * kc2          # the layout regularises; the window can open further
    if masks is None:
        lam1 = ((MU0 / 2) * np.exp(-kc1 * d1_um)) ** 2
        lam2 = ((MU0 / 2) * np.exp(-kc2 * d2_um)) ** 2
        g1, g2 = _tikhonov_pair(bz_t, dx_um, d1_um, d2_um, lam1, lam2)
    else:
        m1, m2 = masks
        g1 = stream_from_bz(bz_t, dx_um, d1_um, kc1) * m1
        g2 = np.zeros_like(bz_t)
        for _ in range(n_iter):
            g2 = stream_from_bz(bz_t - _forward_stream(g1, dx_um, d1_um), dx_um, d2_um, kc2) * m2
            g1 = stream_from_bz(bz_t - _forward_stream(g2, dx_um, d2_um), dx_um, d1_um, kc1) * m1
    return currents_from_stream(g1, dx_um), currents_from_stream(g2, dx_um), (g1, g2)


def layout_mask(g_true_a, dx_um, halo_um: float = 2.0):
    """Where a metal level has wires, from its design: the band around the loop edges (where the
    stream function changes), widened by a halo to allow for registration error. The loop's
    interior is included so that the stream function may hold its plateau there."""
    from scipy.ndimage import binary_dilation
    it = max(1, int(round(halo_um / dx_um)))
    return binary_dilation(g_true_a > 0.02 * g_true_a.max(), iterations=it)


def loop_current_a(g_a, inside_mask):
    """Current of a loop from its stream function: the plateau inside minus the level outside."""
    return float(np.median(g_a[inside_mask]) - np.median(g_a[~inside_mask]))


def fit_layout_currents(bz_t, dx_um, templates):
    """The failure-analysis form of the problem: the current paths are known from the design and
    only the current in each net is unknown. `templates` = [(unit stream function, depth_um), ...]
    for 1 A in each path; the amplitudes follow from linear least squares on the field map."""
    cols = [_forward_stream(g, dx_um, d).ravel() for g, d in templates]
    A = np.stack(cols, axis=1)
    amps, *_ = np.linalg.lstsq(A, bz_t.ravel(), rcond=None)
    return amps
