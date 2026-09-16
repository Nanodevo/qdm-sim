"""Currents as three-dimensional paths: straight segments, vias included, and the full field
vector they produce at the NV plane; the vector as the four NV orientations measure it; and
the fit of net currents to the vector field.

A vertical segment on its own produces a purely azimuthal field, no Bz at all. But a segment
on its own is not a current: in a closed circuit the via is written into Bz as the end of one
trace and the start of another at a different depth. Above all sources the field is a potential
field, so Bz alone fixes the in-plane components (hilbert_inplane); the four NV orientations
therefore add redundancy against noise and information at the edges of a finite field of view,
not a new view of vias. The model checks that claim numerically (hilbert_residual).
"""
from __future__ import annotations

import numpy as np

from .nv import NV_AXES

MU0 = 4e-7 * np.pi


def segment_field(a, b, i_a, X, Y, z_obs):
    """Field (Bx, By, Bz) of a finite straight segment a -> b carrying i_a amperes, on the grid
    (X, Y) at height z_obs. Closed form of Biot-Savart for a straight wire."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    l = b - a
    L = np.linalg.norm(l)
    u = l / L
    rx, ry, rz = X - a[0], Y - a[1], z_obs - a[2]
    t = rx * u[0] + ry * u[1] + rz * u[2]
    px, py, pz = rx - t * u[0], ry - t * u[1], rz - t * u[2]
    rho = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    c1 = t / np.sqrt(t ** 2 + rho ** 2)
    c2 = (t - L) / np.sqrt((t - L) ** 2 + rho ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = MU0 * i_a / (4 * np.pi * rho) * (c1 - c2)
        phix, phiy, phiz = (u[1] * pz - u[2] * py) / rho, (u[2] * px - u[0] * pz) / rho, (u[0] * py - u[1] * px) / rho
    mag = np.where(rho > 1e-12, mag, 0.0)
    return (np.nan_to_num(mag * phix), np.nan_to_num(mag * phiy), np.nan_to_num(mag * phiz))


def path_field(points, i_a, X, Y, z_obs):
    """Field vector of a polyline (list of 3D points, metres) carrying i_a."""
    bx = np.zeros_like(X); by = np.zeros_like(X); bz = np.zeros_like(X)
    for p, q in zip(points[:-1], points[1:]):
        fx, fy, fz = segment_field(p, q, i_a, X, Y, z_obs)
        bx += fx; by += fy; bz += fz
    return bx, by, bz


def connectivity_scene(d1_um=1.0, d2_um=4.0):
    """Two nets that share entry and exit: net A dives through a via to the deep metal M1 and
    comes back up; net B stays in the top metal M3 alongside. Coordinates in metres, z = 0 is the
    NV layer, metals at -d1 (M3) and -d2 (M1). Points are offset in y so the paths are distinct."""
    u = 1e-6
    z3, z1 = -d1_um * u, -d2_um * u
    # the pads sit well outside the 64 um field of view, so that inside it every wire end is a via and
    # not a place where current appears from nowhere (which the vertical signature would also flag)
    net_a = [(-90 * u, -4 * u, z3), (-6 * u, -4 * u, z3), (-6 * u, -4 * u, z1), (14 * u, -4 * u, z1), (14 * u, -4 * u, z3), (90 * u, -4 * u, z3)]
    net_b = [(-90 * u, 6 * u, z3), (90 * u, 6 * u, z3)]
    return {"A: M3 → via → M1 → via → M3": net_a, "B: M3 straight": net_b}


def scene_fields(nets, currents_a, X, Y, z_obs=0.0):
    bx = np.zeros_like(X); by = np.zeros_like(X); bz = np.zeros_like(X)
    for (name, pts), i in zip(nets.items(), currents_a):
        fx, fy, fz = path_field(pts, i, X, Y, z_obs)
        bx += fx; by += fy; bz += fz
    return bx, by, bz


# ---------------------------------------------------------------- what the NVs measure
def projections(bx, by, bz, axes=NV_AXES):
    """The field component along each NV axis: what each orientation's line shift reports."""
    return [bx * ax[0] + by * ax[1] + bz * ax[2] for ax in axes]


def vector_from_projections(proj, axes=NV_AXES):
    """Least-squares vector field from the four projections (three unknowns, four equations)."""
    A = np.asarray(axes, float)
    pinv = np.linalg.pinv(A)                      # 3 x 4
    P = np.stack([p.ravel() for p in proj], axis=0)  # 4 x n
    B = pinv @ P
    return tuple(B[i].reshape(proj[0].shape) for i in range(3))


def fit_net_currents(measured, templates):
    """Net currents from a measured field (tuple of component maps) given each net's field for
    1 A (tuple of the same components). Works with (bz,) alone or (bx, by, bz)."""
    y = np.concatenate([m.ravel() for m in measured])
    A = np.stack([np.concatenate([t.ravel() for t in tpl]) for tpl in templates], axis=1)
    amps, *_ = np.linalg.lstsq(A, y, rcond=None)
    return amps


def hilbert_inplane(bz, dx_m):
    """The in-plane field that Bz implies above all sources: B is then a potential field whose
    Fourier components decay upward as exp(-kz), so Bx(k) = -i kx/k Bz(k) and By(k) = -i ky/k Bz(k).
    This holds for any closed current distribution below the plane, vias included."""
    n = bz.shape[0]
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=dx_m)
    kx, ky = np.meshgrid(k1, k1, indexing="xy")
    k = np.hypot(kx, ky)
    B = np.fft.fft2(bz)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bx = -1j * kx / k * B
        By = -1j * ky / k * B
    Bx[k == 0] = 0; By[k == 0] = 0
    return np.real(np.fft.ifft2(Bx)), np.real(np.fft.ifft2(By))


def bz_from_projection(proj, axis, dx_m):
    """Bz from the projection of the field on one NV axis, on a plane above all sources. With
    Bx = -i kx/k Bz and By = -i ky/k Bz (hilbert_inplane) the projection n.B has the Fourier
    transform P(k) = (n_z - i (n_x kx + n_y ky)/k) Bz(k): a fixed filter whose magnitude is never
    below |n_z|, so a single orientation's map, e.g. one <111> axis on a (100) plate where
    n_z = 0.577, can be turned into a Bz map before the inversion. The k = 0 term, where the
    in-plane relation is undefined, is taken as P / n_z."""
    n = proj.shape[0]
    nx, ny, nz = np.asarray(axis, float) / np.linalg.norm(axis)
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=dx_m)
    kx, ky = np.meshgrid(k1, k1, indexing="xy")
    k = np.hypot(kx, ky)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = nz - 1j * (nx * kx + ny * ky) / k
    H[k == 0] = nz
    return np.real(np.fft.ifft2(np.fft.fft2(proj) / H))


def hilbert_residual(bx, by, bz, dx_m):
    """|measured in-plane field - in-plane field predicted from Bz|. For a closed circuit it is a
    finite-window artifact that shrinks as the window grows (0.2 % of the field at +-128 um for
    the scene of figure 8); it is not a via detector, which was this model's first, wrong idea."""
    px, py = hilbert_inplane(bz, dx_m)
    return np.hypot(bx - px, by - py)
