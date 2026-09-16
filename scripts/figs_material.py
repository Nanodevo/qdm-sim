"""Figures 5 and 6: the diamond sensor as material parameters, and the measurement protocols."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdm import material, photons, protocols

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
                     "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
COLL = photons.collection_fraction(0.9)      # bare diamond, NA 0.9
PIX, LAYER, I0 = 0.5, 1.0, 1e3

# ------------------------------------------------------------------ fig 5: material
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.4))
a = ax[0, 0]
dose = np.logspace(-2, 1.3, 200)
for n in (1, 10, 50):
    a.loglog(dose, [material.nv_density_cm3(n, d) for d in dose], label=f"[N] = {n} ppm")
a.set(title="(a) NV⁻ density vs electron-irradiation dose (10 % yield ceiling)", xlabel="dose (10¹⁸ e⁻/cm²)", ylabel="NV⁻ density (cm⁻³)")
a.legend(loc="lower right")

a = ax[0, 1]
nppm = np.logspace(-1, 2.3, 200)
for x, lab in ((material.C13_NATURAL, "natural ¹³C (1.1 %)"), (1e-4, "¹²C-enriched (0.01 % ¹³C)")):
    a.loglog(nppm, [1e6 * material.t2_star_s(n, x) for n in nppm], label=lab)
a.set(title="(b) ensemble T₂* vs nitrogen content", xlabel="[N] (ppm)", ylabel="T₂* (µs)")
lo, hi = a.get_ylim()
a2 = a.twinx(); a2.set_yscale("log")
a2.set_ylim(1 / (np.pi * lo * 1e-6) / 1e6, 1 / (np.pi * hi * 1e-6) / 1e6)   # inverted: a long T2* is a narrow line
a2.set_ylabel("linewidth 1/(π T₂*) (MHz)"); a2.grid(False)
a.legend(loc="upper right")

a = ax[1, 0]
for x, lab in ((material.C13_NATURAL, "natural ¹³C"), (1e-4, "¹²C-enriched")):
    eta = []
    for n in nppm:
        dens = material.nv_density_cm3(n, 2.0)
        r = photons.photon_rate_per_pixel(I0, dens, LAYER, PIX, COLL)
        w = material.linewidth_hz(material.t2_star_s(n, x))
        eta.append(1e9 * protocols.eta_cw(r, w, material.contrast(), 2.0))
    a.loglog(nppm, eta, label=lab)
    i = int(np.argmin(eta)); a.plot(nppm[i], eta[i], "ko", ms=4); a.text(nppm[i] * 1.15, eta[i] * 0.85, f"optimum ≈ {nppm[i]:.1f} ppm", fontsize=8)
a.set(title="(c) CW sensitivity per 0.5 µm pixel vs [N]: more NVs against shorter T₂*", xlabel="[N] (ppm)", ylabel="sensitivity (nT/√Hz)")
a.legend(loc="upper right")

a = ax[1, 1]
thick = np.linspace(0.005, 0.3, 300)
for surf in ("oxygen", "hydrogen"):
    a.plot(thick * 1e3, [100 * material.active_layer_um(t, surf) / t for t in thick], label=f"{surf}-terminated surface")
a.set(title="(d) share of a shallow NV layer that stays NV⁻", xlabel="NV layer thickness (nm)", ylabel="active fraction (%)", ylim=(0, 105))
a.legend(loc="lower right")
fig.suptitle("The diamond as a sensor: growth, irradiation, isotopes and surface, expressed as density, coherence and contrast", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig3_material.png"); print("fig3")

# ------------------------------------------------------------------ fig 6: protocols
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
R = photons.photon_rate_per_pixel(I0, 1e17, LAYER, PIX, COLL)
w0, c0 = 1e6 / 3, 0.02                         # T2* = 1 us line
a = ax[0]
s = np.logspace(-1.3, 1.5, 200)
w = np.array([protocols.cw_line(v, w0, c0)[0] for v in s]) / w0
c = np.array([protocols.cw_line(v, w0, c0)[1] for v in s]) / c0
eta = np.array([protocols.eta_cw(R, w0, c0, v) for v in s])
a.semilogx(s, w, label="linewidth / intrinsic")
a.semilogx(s, c, label="contrast / maximum")
a.semilogx(s, eta / eta.min(), label="sensitivity / best")
a.axvline(2, color="k", ls=":", lw=0.8); a.text(2.2, 3.2, "optimum s = 2:\nwidth ×√3, contrast ⅔", fontsize=8)
a.set(title="(a) CW ODMR against microwave drive", xlabel="saturation parameter s = (Ω/Ω_sat)²", ylabel="relative", ylim=(0, 4))
a.legend(loc="upper left")

a = ax[1]
t2 = np.logspace(-7, -4.3, 200)
a.loglog(t2 * 1e6, [1e9 * protocols.eta_cw(R, material.linewidth_hz(t), c0, 2.0) for t in t2], label="CW ODMR, optimal drive")
a.loglog(t2 * 1e6, [1e9 * protocols.eta_pulsed_odmr(R, t, c0) for t in t2], label="pulsed ODMR (1 µs readout, 2 µs init)")
a.loglog(t2 * 1e6, [1e9 * protocols.eta_ramsey(R, t, c0) for t in t2], label="Ramsey, τ = T₂*/2")
a.set(title="(b) sensitivity vs T₂* at the widefield photon rate", xlabel="T₂* (µs)", ylabel="sensitivity per pixel (nT/√Hz)")
a.legend(loc="upper right")

a = ax[2]
tr = np.logspace(-7, -5, 200)
T2 = 5e-6
a.semilogx(tr * 1e6, [1e9 * protocols.eta_pulsed_odmr(R, T2, c0, t_read_s=t) for t in tr], label="pulsed ODMR")
a.semilogx(tr * 1e6, [1e9 * protocols.eta_ramsey(R, T2, c0, t_read_s=t) for t in tr], label="Ramsey")
a.axhline(1e9 * protocols.eta_cw(R, material.linewidth_hz(T2), c0, 2.0), color="k", ls="--", lw=0.8, label="CW ODMR at the same T₂*")
a.set(title="(c) the readout duty cycle (T₂* = 5 µs, 2 µs init)", xlabel="readout window (µs)", ylabel="sensitivity per pixel (nT/√Hz)")
a.legend(loc="upper right")
fig.suptitle("Protocols: what sets the frequency resolution, and what the duty cycle costs in photons", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig4_protocols.png"); print("fig4")
