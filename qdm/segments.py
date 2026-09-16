"""Currents as three-dimensional paths: straight segments, vias included, and the full field
vector they produce at the NV plane; the vector as the four NV orientations measure it; and
the fit of net currents to the vector field.

A vertical segment produces a purely azimuthal field: no Bz at all. That is why a via is
invisible to a Bz-only map and why the vector matters for packaging problems.
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
    net_a = [(-30 * u, -4 * u, z3), (-6 * u, -4 * u, z3), (-6 * u, -4 * u, z1), (14 * u, -4 * u, z1), (14 * u, -4 * u, z3), (30 * u, -4 * u, z3)]
    net_b = [(-30 * u, 6 * u, z3), (30 * u, 6 * u, z3)]
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
