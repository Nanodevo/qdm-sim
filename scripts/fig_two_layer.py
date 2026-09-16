"""Figure 7: two current layers at known depths, from one field map, with three levels of prior knowledge."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdm import current

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5, "figure.dpi": 130})
n, dx = 128, 0.5
x, y = current.grid(n, dx)
D1, D2 = 1.0, 4.0
g1 = current.rounded_rect_stream(x, y, -6, -4, 13, 8, 20e-6, 0.5)
g2 = current.rounded_rect_stream(x, y, 7, 6, 17, 12, 60e-6, 0.5)
j1, j2 = current.currents_from_stream(g1, dx), current.currents_from_stream(g2, dx)
bz = current.bz_from_sheet(*j1, dx, D1) + current.bz_from_sheet(*j2, dx, D2)
rng = np.random.default_rng(3)
SIG = 0.22e-6                                   # solid immersion lens, 1 s per pixel
bz_n = bz + rng.normal(0, SIG, bz.shape)
snr = np.abs(bz).max() / SIG
masks = (current.layout_mask(g1, dx), current.layout_mask(g2, dx))
in1, in2 = g1 > 0.5 * g1.max(), g2 > 0.5 * g2.max()

single = current.reconstruct_currents(bz_n, dx, D1, k_cut_per_um=np.log(snr) / D1)
(s1, s2) = current.reconstruct_two_layers(bz_n, dx, D1, D2, snr)[:2]
l1, l2, (gl1, gl2) = current.reconstruct_two_layers(bz_n, dx, D1, D2, snr, masks=masks)
ext = [x[0], x[-1], y[0], y[-1]]
jmax = np.hypot(*j1).max()

fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.8))
def show(a, j, title, vmax=jmax):
    im = a.imshow(np.hypot(*j), extent=ext, origin="lower", cmap="magma", vmin=0, vmax=vmax); a.set_title(title, fontsize=10); a.set(xlabel="x (µm)", ylabel="y (µm)")
    a.contour(x, y, masks[0], levels=[0.5], colors="#3ddc84", linewidths=0.6); a.contour(x, y, masks[1], levels=[0.5], colors="#5bc8ff", linewidths=0.6)
    return im
show(ax[0, 0], single, "(a) one depth assumed (1 µm): both loops in one map")
show(ax[0, 1], s1, "(b) spectral split, layer 1: only what is sharp")
show(ax[0, 2], s2, "(c) spectral split, layer 2: shares the smooth part")
show(ax[1, 0], l1, "(d) layout-constrained, layer 1 (green outline = allowed)")
show(ax[1, 1], l2, "(e) layout-constrained, layer 2 (blue outline = allowed)")
a = ax[1, 2]
sigmas = np.array([0.22, 0.67, 2.7]) * 1e-6; labels = ["SIL, 1 s", "bare, 1 s", "bare, 0.1 s"]
rows = {"spectral split": [], "layout-constrained": [], "template fit (paths known)": []}
u1, u2 = g1 / 20e-6, g2 / 60e-6
for s in sigmas:
    b = bz + rng.normal(0, s, bz.shape); q = np.abs(bz).max() / s
    _, _, (a1, a2) = current.reconstruct_two_layers(b, dx, D1, D2, q)
    _, _, (m1, m2) = current.reconstruct_two_layers(b, dx, D1, D2, q, masks=masks)
    amps = current.fit_layout_currents(b, dx, [(u1, D1), (u2, D2)])
    rows["spectral split"].append((current.loop_current_a(a1, in1), current.loop_current_a(a2, in2)))
    rows["layout-constrained"].append((current.loop_current_a(m1, in1), current.loop_current_a(m2, in2)))
    rows["template fit (paths known)"].append(tuple(amps))
w = 0.25; xs = np.arange(len(sigmas))
for i, (lab, vals) in enumerate(rows.items()):
    v = np.array(vals) * 1e6
    a.bar(xs + (i - 1) * w, v[:, 1], w, label=f"{lab}: deep loop", color=f"C{i}")
    a.bar(xs + (i - 1) * w, v[:, 0], w * 0.5, color=f"C{i}", alpha=0.55, hatch="//")
a.axhline(60, color="k", ls="--", lw=0.8, label="true 60 µA (deep)"); a.axhline(20, color="k", ls=":", lw=0.8, label="true 20 µA (shallow, hatched bars)")
a.set(title="(f) recovered loop currents against noise", xticks=xs, xticklabels=labels, ylabel="current (µA)", ylim=(0, 92))
a.set_ylim(0, 100); a.legend(loc="upper left", fontsize=7.5, ncol=1, framealpha=0.95)
fig.suptitle("Two metal levels from one field map: what the depths and the layout buy (SIL, 1 s per pixel, 219 nT noise)", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig8_two_layer.png"); print("fig8")
for lab, vals in rows.items():
    print(lab, [(round(a * 1e6, 1), round(b * 1e6, 1)) for a, b in vals])
