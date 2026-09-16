"""All figures of the study. Run: python scripts/figs.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdm import current, nv, optics, photons

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
                     "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})

B_DIR = np.array([0.3, 0.5, 0.81]); B_DIR /= np.linalg.norm(B_DIR)

# ------------------------------------------------------------------ fig 1: spin physics
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
a = ax[0]
bs = np.linspace(0, 10e-3, 200)
lines = np.array([nv.all_transitions_ghz(b * B_DIR).ravel() for b in bs])
for i in range(8):
    a.plot(bs * 1e3, lines[:, i], color=f"C{i // 2}", lw=1.2)
a.set(title="(a) the eight ODMR lines vs field", xlabel="|B| (mT)", ylabel="transition frequency (GHz)")
a.text(0.02, 0.97, "one colour per NV orientation,\ntwo lines each", transform=a.transAxes, va="top", fontsize=8.5)

a = ax[1]
f = np.linspace(2.78, 2.96, 4000)
b = 3e-3 * B_DIR
a.plot(f, nv.odmr_spectrum(f, b, contrast=0.02, linewidth_mhz=1.0), lw=1)
ln = nv.all_transitions_ghz(b)
a.annotate("", xy=(ln[0, 0], 1.00035), xytext=(ln[0, 1], 1.00035), arrowprops=dict(arrowstyle="<->", color="C0"))
a.text((ln[0, 0] + ln[0, 1]) / 2, 1.00045, "2γB∥ for the orientation most aligned with B", ha="center", fontsize=8.5, color="C0")
a.set(title="(b) ensemble ODMR at 3 mT (2 % contrast, 1 MHz lines)", xlabel="microwave frequency (GHz)", ylabel="fluorescence (normalised)", ylim=(0.9946, 1.0008))

a = ax[2]
f0 = ln[0, 1]
fz = np.linspace(f0 - 4e-3, f0 + 4e-3, 800)
s0 = nv.odmr_spectrum(fz, b, 0.02, 1.0)
s1 = nv.odmr_spectrum(fz, b + 20e-6 * B_DIR, 0.02, 1.0)
a.plot((fz - f0) * 1e3, s0, label="B")
a.plot((fz - f0) * 1e3, s1, label="B + 20 µT along the axis: shift 0.56 MHz")
a.axvline(-0.29, color="k", ls=":", lw=0.8)
a.annotate("readout point:\nthe steepest slope", xy=(-0.29, 0.9975), xytext=(-3.8, 0.9962), fontsize=8, arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
a.set(title="(c) how a field becomes a signal", xlabel="detuning from the line (MHz)", ylabel="fluorescence (normalised)")
a.set_ylim(0.9925, 1.00025); a.legend(loc="lower right")
fig.suptitle("Nitrogen-vacancy ground state: D = 2.870 GHz, γ = 28.0 MHz/mT, four <111> orientations", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig1_odmr.png"); print("fig1")

# ------------------------------------------------------------------ fig 2: photons and sensitivity
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
na = np.linspace(0.1, 0.95, 200)
a = ax[0]
a.plot(na, [100 * photons.collection_fraction(v) for v in na], label="bare (100) surface")
a.plot(na, [100 * photons.collection_fraction(v, back_mirror=True) for v in na], label="bare, mirrored back side")
a.plot(na, [100 * photons.collection_fraction(v, sil=True) for v in na], label="solid immersion lens")
a.axhline(100 * photons.escape_cone_fraction() * photons.FRESNEL_T_NORMAL, color="k", ls="--", lw=0.8, label="flat-surface escape cone: 3.8 %")
a.set(title="(a) fraction of NV photons collected", xlabel="numerical aperture", ylabel="collected (%)", yscale="log", ylim=(0.05, 60))
a.set_ylim(0.015, 60); a.legend(loc="lower right", fontsize=8)

a = ax[1]
I = np.logspace(1, 6.3, 300)
cfg = {"bare, NA 0.9": photons.collection_fraction(0.9), "back mirror, NA 0.9": photons.collection_fraction(0.9, back_mirror=True), "SIL, NA 0.9": photons.collection_fraction(0.9, sil=True)}
for lab, c in cfg.items():
    a.loglog(I, [photons.photon_rate_per_pixel(i, 1e17, 1.0, 1.0, c) for i in I], label=lab)
a.axvline(1e3, color="gray", ls=":", lw=0.8); a.text(7e2, 3e10, "typical widefield\n1 kW/cm²", fontsize=8, color="gray", ha="right")
a.axhline(1e7, color="gray", ls="--", lw=0.8); a.text(2e6, 1.4e7, "camera full well at 100 frames/s", fontsize=8, color="gray", ha="right")
a.set(title="(b) detected photons per 1 µm pixel", xlabel="532 nm intensity (W/cm²)", ylabel="photons / s / pixel")
a.legend(loc="lower right")

a = ax[2]
coll = np.logspace(-3, np.log10(0.6), 200)
for t_lab, t in (("1 s per pixel", 1.0), ("10 s per pixel", 10.0)):
    eta = [photons.field_noise(photons.sensitivity_t_per_rthz(photons.photon_rate_per_pixel(1e3, 1e17, 1.0, 1.0, c)), t) * 1e9 for c in coll]
    a.loglog(coll * 100, eta, label=t_lab)
for lab, c in cfg.items():
    r = photons.photon_rate_per_pixel(1e3, 1e17, 1.0, 1.0, c)
    e = photons.field_noise(photons.sensitivity_t_per_rthz(r), 1.0) * 1e9
    a.plot(c * 100, e, "o", color="k", ms=4); a.text(c * 100 * 1.1, e * 1.1, lab, fontsize=8)
a.set(title="(c) field noise per pixel vs collection", xlabel="collected fraction (%)", ylabel="field noise per pixel (nT)")
a.legend(loc="upper right")
fig.suptitle("From illumination to photons to sensitivity: the collection efficiency enters as its square root", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig2_photons.png"); print("fig2")

# ------------------------------------------------------------------ fig 3: resolution
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
a = ax[0]
x = np.linspace(-12, 12, 1200)
for d in (0.5, 1.0, 2.0, 4.0):
    a.plot(x, optics.wire_bz(x, 1e-3, d) * 1e6, label=f"stand-off {d} µm")
a.set(title="(a) field of a 1 mA wire at the NV plane", xlabel="x (µm)", ylabel="Bz (µT)")
a.legend(loc="upper left")

a = ax[1]
ds = np.linspace(0.1, 6, 200)
for n_a in (0.5, 0.9):
    a.plot(ds, [optics.wire_response_width_um(n_a, d) for d in ds], label=f"NA {n_a}: peak-to-peak width")
    a.axhline(optics.diffraction_fwhm_um(n_a), color="gray", ls=":", lw=0.8)
a.plot(ds, 2 * ds, "k--", lw=0.8, label="2 × stand-off")
a.text(3.2, 1.15, "diffraction floors 0.71 / 0.40 µm", fontsize=8, color="gray")
a.set(title="(b) imaged wire width vs stand-off", xlabel="stand-off d (µm)", ylabel="width (µm)")
a.legend(loc="upper left")

a = ax[2]
period = np.logspace(-0.3, 2, 300)
k = 2 * np.pi / period
for d in (0.5, 1.0, 2.0, 4.0):
    a.semilogx(period, optics.standoff_transfer(k, d), label=f"d = {d} µm")
a.axhline(0.5, color="k", ls=":", lw=0.8)
a.set(title="(c) stand-off as a low-pass filter", xlabel="spatial period of the current pattern (µm)", ylabel="field amplitude transferred")
a.legend(loc="upper left")
fig.suptitle("Spatial resolution: the NV layer sees the chip through the distance between them", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig5_resolution.png"); print("fig5")

# ------------------------------------------------------------------ fig 4: current map reconstruction
n, dx = 192, 0.5
x, y = current.grid(n, dx)
D_TOP, D_DEEP = 1.0, 4.0
g_top = current.rounded_rect_stream(x, y, -12, -8, 26, 16, 20e-6, 0.5)
g_deep = current.rounded_rect_stream(x, y, 14, 12, 34, 24, 60e-6, 0.5)
jx_t, jy_t = current.currents_from_stream(g_top, dx)
jx_d, jy_d = current.currents_from_stream(g_deep, dx)
bz = current.bz_from_sheet(jx_t, jy_t, dx, D_TOP) + current.bz_from_sheet(jx_d, jy_d, dx, D_DEEP)
rng = np.random.default_rng(7)

def noise_sigma(coll, t):
    r = photons.photon_rate_per_pixel(1e3, 1e17, 1.0, dx, coll)
    return photons.field_noise(photons.sensitivity_t_per_rthz(r), t)

c_bare, c_sil = photons.collection_fraction(0.9), photons.collection_fraction(0.9, sil=True)
s_bare, s_sil = noise_sigma(c_bare, 1.0), noise_sigma(c_sil, 1.0)
bz_bare = bz + rng.normal(0, s_bare, bz.shape)
bz_sil = bz + rng.normal(0, s_sil, bz.shape)
jr_bare = current.reconstruct_currents(bz_bare, dx, D_TOP, k_cut_per_um=2.0 / D_TOP)
jr_sil = current.reconstruct_currents(bz_sil, dx, D_TOP, k_cut_per_um=2.0 / D_TOP)
ext = [x[0], x[-1], y[0], y[-1]]
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.6))
jt = np.hypot(jx_t + jx_d, jy_t + jy_d)
im = ax[0, 0].imshow(jt, extent=ext, origin="lower", cmap="magma"); ax[0, 0].set_title("(a) chip currents: 20 µA loop 1 µm below the NVs,\n60 µA loop 4 µm below")
fig.colorbar(im, ax=ax[0, 0], label="|J| (A/m)")
vmax = np.abs(bz).max() * 1e6
im = ax[0, 1].imshow(bz * 1e6, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0, 1].set_title("(b) Bz at the NV plane, noise-free")
fig.colorbar(im, ax=ax[0, 1], label="Bz (µT)")
im = ax[0, 2].imshow(bz_bare * 1e6, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0, 2].set_title(f"(c) measured: bare diamond, NA 0.9, 1 s per pixel\nnoise {s_bare * 1e9:.0f} nT per 0.5 µm pixel")
fig.colorbar(im, ax=ax[0, 2], label="Bz (µT)")
jr_free = current.reconstruct_currents(bz, dx, D_TOP, k_cut_per_um=2.0 / D_TOP)
jmax = np.hypot(*jr_free).max()
im = ax[1, 0].imshow(np.hypot(*jr_bare), extent=ext, origin="lower", cmap="magma", vmin=0, vmax=jmax); ax[1, 0].set_title("(d) current map from (c),\ninverted at 1 µm")
fig.colorbar(im, ax=ax[1, 0], label="|J| (A/m)")
im = ax[1, 1].imshow(np.hypot(*jr_sil), extent=ext, origin="lower", cmap="magma", vmin=0, vmax=jmax); ax[1, 1].set_title(f"(e) with a solid immersion lens:\nnoise {s_sil * 1e9:.0f} nT")
fig.colorbar(im, ax=ax[1, 1], label="|J| (A/m)")
a = ax[1, 2]
colls = np.logspace(-3, np.log10(0.6), 14)
b_top = current.bz_from_sheet(jx_t, jy_t, dx, D_TOP)
j_ref = current.reconstruct_currents(b_top, dx, D_TOP, k_cut_per_um=2.0 / D_TOP)
for t, lab in ((1.0, "1 s per pixel"), (10.0, "10 s per pixel")):
    errs = []
    for c in colls:
        sig = noise_sigma(c, t)
        jr = current.reconstruct_currents(b_top + rng.normal(0, sig, bz.shape), dx, D_TOP, k_cut_per_um=2.0 / D_TOP)
        errs.append(current.rms_error(jr, j_ref))
    a.semilogx(colls * 100, errs, "o-", ms=3, label=lab)
for c, lab in ((c_bare, "bare"), (c_sil, "SIL")):
    a.axvline(c * 100, color="gray", ls=":", lw=0.8); a.text(c * 100 * 1.1, 0.5, lab, fontsize=8, color="gray", rotation=90, va="center")
a.set(title="(f) noise in the current map vs collection", xlabel="collected fraction (%)", ylabel="RMS error vs noise-free inversion", ylim=(0, 1.0))
a.legend(loc="upper right")
for aa in ax.ravel()[:5]:
    aa.set(xlabel="x (µm)", ylabel="y (µm)"); aa.grid(False)
fig.suptitle("From the chip's currents to the field the NVs see and back: the deep loop is blurred by its stand-off, the shallow one by the photon budget", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig6_reconstruction.png"); print("fig6")
print(f"noise bare {s_bare*1e9:.0f} nT, SIL {s_sil*1e9:.0f} nT; peak Bz {vmax:.2f} uT; collection bare {c_bare*100:.2f} % SIL {c_sil*100:.1f} %")
