"""Figure 8: vertical currents (vias) and why the field vector is needed to see them."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdm import segments

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5, "figure.dpi": 130})
n, dx = 128, 0.5
ax_um = (np.arange(n) - n // 2) * dx
X, Y = np.meshgrid(ax_um * 1e-6, ax_um * 1e-6)
nets = segments.connectivity_scene(1.0, 4.0)
names = list(nets)
GOOD, OPEN = (25e-6, 25e-6), (0.0, 50e-6)          # good chip: the current splits; open via: all of it stays in M3
SIG = 0.22e-6                                        # per projection, SIL and 1 s
rng = np.random.default_rng(5)

def measure(currents):
    bx, by, bz = segments.scene_fields(nets, currents, X, Y)
    proj = [p + rng.normal(0, SIG, p.shape) for p in segments.projections(bx, by, bz)]
    return segments.vector_from_projections(proj), (bx, by, bz)

(gx, gy, gz), (tx, ty, tz) = measure(GOOD)
(ox, oy, oz), _ = measure(OPEN)
tpl = [segments.path_field(pts, 1.0, X, Y, 0.0) for pts in nets.values()]
ext = [ax_um[0], ax_um[-1], ax_um[0], ax_um[-1]]
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.8))
a = ax[0, 0]
for (name, pts), col in zip(nets.items(), ("C1", "C0")):
    P = np.array(pts) * 1e6
    for p, q in zip(P[:-1], P[1:]):
        if abs(p[2] - q[2]) > 1e-9: a.plot(p[0], p[1], "o", color="k", ms=7, mfc="w", mew=1.5)
        else: a.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=4 if p[2] < -2e-6 * 1e6 else 2.5, alpha=0.9)
a.plot([], [], color="C1", lw=2.5, label="net A: M3 (thin) and M1 (thick)"); a.plot([], [], color="C0", lw=2.5, label="net B: M3 only"); a.plot([], [], "o", color="k", mfc="w", label="via, 3 µm vertical")
a.set(title="(a) two nets between pads outside the view; top view", xlim=ext[:2], ylim=ext[2:], xlabel="x (µm)", ylabel="y (µm)"); a.set_aspect("equal"); a.legend(loc="upper left", fontsize=7.5)
vm = np.abs(gz).max() * 1e6
def show(a, m, title, cmap="RdBu_r", vmax=None):
    im = a.imshow(m * 1e6, extent=ext, origin="lower", cmap=cmap, vmin=-vmax if cmap == "RdBu_r" else 0, vmax=vmax); a.set_title(title, fontsize=10); a.set(xlabel="x (µm)", ylabel="y (µm)")
    fig.colorbar(im, ax=a, label="µT", fraction=0.046)
show(ax[0, 1], gz, "(b) good chip: Bz from the four NV orientations", vmax=vm)
hx, hy = segments.hilbert_inplane(gz, dx * 1e-6); vmx = np.hypot(tx, ty).max() * 1e6
show(ax[0, 2], np.hypot(gx, gy), "(c) good chip: in-plane |B| measured by the four orientations", cmap="magma", vmax=vmx)
show(ax[1, 0], oz, "(d) open via: Bz, net A's current is gone", vmax=vm)
show(ax[1, 1], np.hypot(hx, hy), "(e) good chip: in-plane |B| computed from Bz alone (eq. 7)", cmap="magma", vmax=vmx)
a = ax[1, 2]
res = {}
for lab, (bxm, bym, bzm) in (("good", (gx, gy, gz)), ("open via", (ox, oy, oz))):
    res[(lab, "Bz only")] = segments.fit_net_currents((bzm,), [(t[2],) for t in tpl])
    res[(lab, "vector")] = segments.fit_net_currents((bxm, bym, bzm), tpl)
w = 0.2; xs = np.array([0, 1])
for i, (lab, truth) in enumerate((("good", GOOD), ("open via", OPEN))):
    for j, meth in enumerate(("Bz only", "vector")):
        v = res[(lab, meth)] * 1e6
        a.bar(xs + (2 * i + j - 1.5) * w, v, w, color=f"C{j}", alpha=0.55 + 0.45 * i, label=f"{lab}, {meth}")
    a.plot(xs + (2 * i - 1) * w, np.array(truth) * 1e6, "k_", ms=22, mew=2)
a.set(title="(f) net currents fitted to the maps (black = truth)", xticks=xs, xticklabels=["net A (via path)", "net B (M3)"], ylabel="current (µA)", ylim=(-5, 62)); a.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Vertical currents: a via shows in Bz as one trace ending and another starting; the field vector adds redundancy, not new information", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig10_vector.png"); print("fig10")
for k, v in res.items(): print(k, np.round(v * 1e6, 1))
