"""Figure 9: ray-traced collection and illumination."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdm import photons, rays

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5, "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
n = rays.N_DIAMOND
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.8))

# (a) rays from an emitter under a flat surface
a = ax[0, 0]
a.fill_between([-3, 3], -1.5, 0, color="#cfe6ff", alpha=.6); a.axhline(0, color="k", lw=0.8)
a.plot(0, -1.0, "o", color="C3"); a.text(0.08, -1.15, "NV", fontsize=8)
th_c = np.arcsin(1 / n)
for th in np.deg2rad(np.arange(-70, 71, 10)):
    x1, y1 = 1.0 * np.tan(th), 0.0
    if abs(th) < th_c:
        tht = np.arcsin(n * np.sin(th)); T = float(rays.fresnel_t(np.cos(th), n, 1.0))
        a.plot([0, x1], [-1, 0], color="C0", lw=0.9); a.plot([x1, x1 + 1.2 * np.sin(tht)], [0, 1.2 * np.cos(tht)], color="C0", lw=0.9, alpha=0.3 + 0.7 * T)
    else:
        a.plot([0, x1], [-1, 0], color="C3", lw=0.9, ls="--"); a.plot([x1, x1 + 1.0 * np.sin(th)], [0, -1.0 * np.cos(th)], color="C3", lw=0.7, ls="--", alpha=.5)
acc = np.arcsin(0.9); a.plot([0, 1.4 * np.sin(acc)], [0, 1.4 * np.cos(acc)], "k:", lw=1); a.plot([0, -1.4 * np.sin(acc)], [0, 1.4 * np.cos(acc)], "k:", lw=1)
a.text(0.9, 0.95, "NA 0.9\nacceptance", fontsize=8)
a.set(title="(a) flat surface: 24.4° escape cone", xlim=(-2.2, 2.2), ylim=(-1.5, 1.5), xlabel="x (a.u.)", ylabel="z (a.u.)"); a.set_aspect("equal"); a.grid(False)

# (b) rays with a hemispherical SIL
a = ax[0, 1]
circ = np.linspace(0, np.pi, 100); a.fill_between(1.3 * np.cos(circ), 0, 1.3 * np.sin(circ), color="#cfe6ff", alpha=.6); a.fill_between([-2.2, 2.2], -0.6, 0, color="#cfe6ff", alpha=.6); a.axhline(0, color="k", lw=0.5, alpha=.4)
a.plot(0, 0, "o", color="C3"); a.text(0.08, -0.2, "NV at the centre", fontsize=8)
for th in np.deg2rad(np.arange(-80, 81, 10)):
    a.plot([0, 1.6 * np.sin(th)], [0, 1.6 * np.cos(th)], color="C0", lw=0.9, alpha=0.9 if abs(th) < acc else 0.25)
a.plot([0, 1.7 * np.sin(acc)], [0, 1.7 * np.cos(acc)], "k:", lw=1); a.plot([0, -1.7 * np.sin(acc)], [0, 1.7 * np.cos(acc)], "k:", lw=1)
a.set(title="(b) solid immersion lens: rays leave radially", xlim=(-2.2, 2.2), ylim=(-0.7, 1.8), xlabel="x (a.u.)", ylabel="z (a.u.)"); a.set_aspect("equal"); a.grid(False)

# (c) collection vs NA: Monte Carlo against the section-3 estimates
a = ax[0, 2]
nas = np.linspace(0.3, 0.95, 14)
for mode, lab, est, kw in (("bare", "bare", lambda v: photons.collection_fraction(v), {}), ("mirror", "mirrored back", lambda v: photons.collection_fraction(v, back_mirror=True), {}), ("sil", "SIL, uncoated", None, {}), ("sil", "SIL, coated", lambda v: photons.collection_fraction(v, sil=True), {"sil_coated": True})):
    mc = [100 * rays.collection_mc(v, mode, n_rays=100_000, **kw) for v in nas]
    ln, = a.semilogy(nas, mc, "o", ms=4, label=f"{lab}, traced")
    if est: a.semilogy(nas, [100 * est(v) for v in nas], "-", color=ln.get_color(), lw=1, alpha=.7, label=f"{lab}, estimate")
a.set(title="(c) collected fraction: traced vs estimated", xlabel="numerical aperture", ylabel="collected (%)", ylim=(0.3, 400)); a.legend(fontsize=7, loc="upper left", ncol=2, columnspacing=1.0)

# (d) uniformity across the field with a SIL
a = ax[1, 0]
xs = np.linspace(0, 150, 16)
for R in (1000, 500, 250):
    a.plot(xs, [100 * rays.collection_mc(0.9, "sil", n_rays=100_000, field_um=x, sil_radius_um=R) for x in xs], "o-", ms=3, label=f"SIL radius {R / 1000:g} mm")
a.axvspan(0, 32, color="gray", alpha=.15); a.text(2, 20.65, "field of view\nof the simulation", fontsize=8, va="bottom")
a.set(title="(d) SIL collection across the field", xlabel="distance from the SIL centre (µm)", ylabel="collected (%)", ylim=(20.5, 24.5)); a.legend(loc="upper right")

# (e) illumination: refraction into the slab
a = ax[1, 1]
f = rays.illumination_footprint(53.6, 36.0, 300.0)
a.fill_between([-250, 250], -300, 0, color="#cfe6ff", alpha=.6); a.axhline(0, color="k", lw=0.8); a.axhline(-300, color="k", lw=0.8)
th_i, th_t = np.deg2rad(53.6), np.deg2rad(f["theta_t_deg"])
for off in (-36, 0, 36):
    x0 = off / np.cos(th_i); y0 = 0
    a.plot([x0 + 260 * np.sin(th_i), x0], [260 * np.cos(th_i), 0], color="C2", lw=1.2, alpha=.9)
    a.plot([x0, x0 - 300 * np.tan(th_t)], [0, -300], color="C2", lw=1.2, alpha=.9)
a.plot([0, -300 * np.tan(th_t)], [0, -300], color="k", ls=":", lw=0.8)
a.text(-245, -30, "diamond, 300 µm", fontsize=8); a.text(-245, -290, "NV layer", fontsize=8)
a.text(-245, 150, f"in air: 53.6°\nin diamond: {f['theta_t_deg']:.1f}°\nFresnel: {100 * f['fresnel_t']:.0f} % enters", fontsize=8)
a.set(title="(e) the beam refracts at the top face", xlabel="x (µm)", ylabel="z (µm)", xlim=(-250, 250), ylim=(-320, 260)); a.set_aspect("equal"); a.grid(False)

# (f) footprint on the NV layer
a = ax[1, 2]
tt = np.linspace(0, 2 * np.pi, 200)
a.plot(36 * np.cos(tt), 36 * np.sin(tt), "--", color="gray", label="beam cross-section, r = 36 µm")
a.plot(f["semi_axes_um"][0] * np.cos(tt), f["shift_um"] + f["semi_axes_um"][1] * np.sin(tt), color="C2", lw=2, label=f"footprint on the NV layer: {f['semi_axes_um'][0]:.0f} × {f['semi_axes_um'][1]:.0f} µm, shifted {f['shift_um']:.0f} µm")
a.plot(0, 0, "k+", ms=8); a.plot(0, f["shift_um"], "+", color="C2", ms=8)
a.text(-95, -80, f"peak intensity {100 * f['peak_factor']:.0f} % of the beam's:\nFresnel loss × stretch", fontsize=8)
a.set(title="(f) footprint on the NV layer", xlabel="x (µm)", ylabel="y (µm, along the tilt)", xlim=(-100, 100), ylim=(-100, 235)); a.set_aspect("equal"); a.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Ray tracing the two optical paths: what the surfaces do to the collection and to the illumination", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig9_rays.png"); print("fig9")
