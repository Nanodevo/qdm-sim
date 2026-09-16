"""Ray tracing of the optical paths, Monte Carlo, with real surfaces.

Collection: photons leave an NV centre in random directions, reach the diamond's exit surface
(a plane, or a hemispherical solid immersion lens), refract there by Snell's law with the
Fresnel transmission of that angle or are totally internally reflected, and count as collected
if their direction in air lies inside the objective's acceptance cone. This replaces the
solid-angle estimates of photons.py with numbers from the geometry.

Illumination: the excitation beam meets the diamond's top face at an angle, refracts, crosses
the slab and lands on the NV layer as a displaced, stretched ellipse; the footprint and the
Fresnel loss follow from the same laws.
"""
from __future__ import annotations

import numpy as np

N_DIAMOND = 2.417


def fresnel_t(cos_i, n1, n2):
    """Unpolarised Fresnel transmission for incidence cosine cos_i from n1 into n2 (0 where TIR)."""
    cos_i = np.asarray(cos_i, float)
    sin_t2 = (n1 / n2) ** 2 * (1 - cos_i ** 2)
    ok = sin_t2 < 1
    cos_t = np.sqrt(np.clip(1 - sin_t2, 0, None))
    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    return np.where(ok, 1 - 0.5 * (rs + rp), 0.0)


def refract(d, n_vec, eta):
    """Snell in 3D: direction d (unit), surface normal n_vec (unit, pointing into the exit medium),
    eta = n_in / n_out. Returns (d_out, ok) with ok False for total internal reflection."""
    cos_i = np.einsum("ij,ij->i", d, n_vec)
    k = 1 - eta ** 2 * (1 - cos_i ** 2)
    ok = k > 0
    cos_t = np.sqrt(np.clip(k, 0, None))
    d_out = eta * d + (cos_t - eta * cos_i)[:, None] * n_vec      # d_t = eta d + (cos_t - eta cos_i) n
    return d_out, ok, cos_i


def collection_mc(na, mode="bare", n_rays=400_000, field_um=0.0, sil_radius_um=1000.0, n=N_DIAMOND,
                  sil_coated=False, seed=0):
    """Collected fraction of an isotropic emitter in the NV layer.

    mode: 'bare' (flat exit surface), 'mirror' (flat surface, ideal reflector under the NV layer),
    'sil' (hemispherical solid immersion lens of radius sil_radius_um centred on the field centre,
    index-matched to the diamond; sil_coated=True treats its surface as anti-reflection coated).
    field_um: lateral position of the emitter, to see how uniform the collection is across the view.
    """
    rng = np.random.default_rng(seed)
    cz = rng.uniform(-1, 1, n_rays)
    ph = rng.uniform(0, 2 * np.pi, n_rays)
    s = np.sqrt(1 - cz ** 2)
    d = np.stack([s * np.cos(ph), s * np.sin(ph), cz], axis=1)
    if mode == "mirror":
        d[:, 2] = np.abs(d[:, 2])                       # the downward half comes back up
    up = d[:, 2] > 0
    d = d[up]
    weight = np.zeros(len(d))
    if mode in ("bare", "mirror"):
        n_vec = np.tile([0.0, 0.0, 1.0], (len(d), 1))
        d_out, ok, cos_i = refract(d, n_vec, n / 1.0)
        t = fresnel_t(cos_i, n, 1.0)
    else:
        R = sil_radius_um
        src = np.array([field_um, 0.0, 0.0])
        sd = d @ src
        tt = -sd + np.sqrt(sd ** 2 - src @ src + R ** 2)
        p = src + tt[:, None] * d
        n_vec = p / R
        d_out, ok, cos_i = refract(d, n_vec, n / 1.0)
        t = np.ones(len(d)) if sil_coated else fresnel_t(cos_i, n, 1.0)
    ang_ok = ok & (d_out[:, 2] > 0) & (d_out[:, 2] >= np.sqrt(1 - na ** 2))   # inside the acceptance cone
    weight[ang_ok] = t[ang_ok]
    return float(weight.sum() / n_rays)


def illumination_footprint(theta_i_deg, w_um, slab_um, n=N_DIAMOND):
    """A circular Gaussian beam of 1/e^2 radius w, incident on the top face at theta_i, inside the
    diamond and on the NV layer a distance slab_um below: the footprint is an ellipse w by
    w / cos(theta_i), displaced along the tilt by slab * tan(theta_t), with the Fresnel loss of
    the entrance and the peak intensity reduced by the stretch."""
    th_i = np.deg2rad(theta_i_deg)
    th_t = np.arcsin(np.sin(th_i) / n)
    t = float(fresnel_t(np.cos(th_i), 1.0, n))
    return dict(theta_t_deg=float(np.rad2deg(th_t)), semi_axes_um=(w_um, w_um / np.cos(th_i)),
                shift_um=float(slab_um * np.tan(th_t)), fresnel_t=t, peak_factor=float(t * np.cos(th_i)))
