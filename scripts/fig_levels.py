"""Figure 1: where the NV- levels sit in diamond's band gap, the optical cycle that makes the spin
readable, and the ground-state spin sublevels the microwave addresses. A schematic: the numbers are
from Aslam et al. 2013 (band positions), Doherty et al. 2013 and Rondin et al. 2014 (level scheme)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
E_GAP, E_GS, E_ZPL, E_SING = 5.47, 2.94, 1.945, 1.19          # eV: band gap, 3A2 above the VBM, 3A2-3E zero-phonon line, 1A1-1E line
E_1E = 0.40                                                    # eV above 3A2: drawn position of the lower singlet (only approximately known)
E_1A1 = E_1E + E_SING
GREEN = 2.33                                                   # eV, 532 nm
C_G, C_R, C_S = "#2e9e4f", "#d1493a", "#7a6fc2"

fig, ax = plt.subplots(1, 3, figsize=(12.5, 5.4))

# (a) band gap
a = ax[0]
a.add_patch(Rectangle((0, -0.6), 1, 0.6, color="#c9ced6")); a.add_patch(Rectangle((0, E_GAP), 1, 0.6, color="#c9ced6"))
a.text(0.5, -0.32, "valence band", ha="center", va="center", fontsize=8.5); a.text(0.5, E_GAP + 0.28, "conduction band", ha="center", va="center", fontsize=8.5)
a.annotate("", xy=(0.06, E_GAP), xytext=(0.06, 0), arrowprops=dict(arrowstyle="<->", lw=0.9)); a.text(0.08, E_GAP / 2, "5.47 eV\nband gap", fontsize=7.5, va="center")
for e, lab, ls, col in ((E_GS, "NV⁻ ³A₂ ground state", "-", "k"), (E_GS + E_ZPL, "³E excited state", "-", "k"), (E_GS + E_1A1, "¹A₁", "--", C_S), (E_GS + E_1E, "¹E", "--", C_S)):
    a.plot([0.42, 0.85], [e, e], color=col, ls=ls, lw=2 if ls == "-" else 1.2); a.text(0.87, e, lab, fontsize=8, va="center", color=col)
a.add_patch(Rectangle((0.42, E_GS + E_ZPL), 0.43, 0.5, color=C_R, alpha=.12)); a.text(0.84, E_GS + E_ZPL + 0.3, "phonon\nsideband", fontsize=7, ha="right", va="center", color=C_R)
a.annotate("", xy=(0.47, E_GS + GREEN), xytext=(0.47, E_GS), arrowprops=dict(arrowstyle="-|>", color=C_G, lw=1.6)); a.text(0.49, E_GS + 0.75, "532 nm\n2.33 eV", fontsize=7.5, color=C_G)
a.annotate("", xy=(0.6, E_GAP), xytext=(0.6, E_GS + E_ZPL), arrowprops=dict(arrowstyle="-|>", color=C_G, lw=1.2, ls="--")); a.text(1.02, E_GS + E_ZPL + 0.16, "a second green photon\nionises the excited\ncentre to NV⁰", fontsize=7, color=C_G, va="bottom")
a.annotate("", xy=(0.28, E_GS), xytext=(0.28, 0), arrowprops=dict(arrowstyle="<->", lw=0.8, color="gray")); a.text(0.30, E_GS / 2 - 0.05, "2.94 eV", fontsize=7.5, color="gray")
a.annotate("", xy=(0.28, E_GAP), xytext=(0.28, E_GS), arrowprops=dict(arrowstyle="<->", lw=0.8, color="gray")); a.text(0.30, E_GS + 1.35, "2.6 eV", fontsize=7.5, color="gray")
a.set(title="(a) the NV⁻ levels in diamond's band gap", xlim=(0, 1.55), ylim=(-0.6, 6.1), ylabel="energy above the valence band (eV)"); a.set_xticks([]); a.grid(False)
for sp in ("top", "right", "bottom"): a.spines[sp].set_visible(False)

# (b) the optical cycle
a = ax[1]
dz = 0.11   # drawn (exaggerated) separation of the spin sublevels
def level(y, x0, x1, lab, col="k", lw=1.6, ls="-"):
    a.plot([x0, x1], [y, y], color=col, lw=lw, ls=ls); a.text(x1 + 0.02, y, lab, fontsize=7.5, va="center", color=col)
level(0, 0.05, 0.45, "m_s = 0"); level(dz, 0.05, 0.45, "m_s = ±1   (D = 2.87 GHz)")
a.text(0.47, dz + 0.24, "³A₂ ground state", fontsize=8.5, fontweight="bold")
level(E_ZPL, 0.05, 0.45, "m_s = 0"); level(E_ZPL + dz, 0.05, 0.45, "m_s = ±1   (1.42 GHz)")
a.text(0.47, E_ZPL + dz + 0.13, "³E excited state, 12 ns", fontsize=8.5, fontweight="bold")
a.add_patch(Rectangle((0.05, E_ZPL + dz + 0.02), 0.4, 0.45, color=C_R, alpha=.12)); a.add_patch(Rectangle((0.05, dz + 0.02), 0.4, 0.45, color=C_R, alpha=.12))
a.text(0.31, E_ZPL + dz + 0.36, "phonon sideband", fontsize=7, ha="center", color=C_R); a.text(0.2, dz + 0.42, "phonon\nsideband", fontsize=7, ha="center", color=C_R)
level(E_1A1, 0.95, 1.25, "¹A₁", C_S, 1.4, "--"); level(E_1E, 0.95, 1.25, "¹E, ~200 ns", C_S, 1.4, "--")
a.annotate("", xy=(0.1, GREEN), xytext=(0.1, 0), arrowprops=dict(arrowstyle="-|>", color=C_G, lw=1.8)); a.text(0.115, 1.05, "532 nm\nexcitation", fontsize=7.5, color=C_G)
a.annotate("", xy=(0.33, dz + 0.3), xytext=(0.33, E_ZPL), arrowprops=dict(arrowstyle="-|>", color=C_R, lw=1.8)); a.text(0.345, 0.95, "fluorescence\n637 nm line\n+ sideband\nto 800 nm", fontsize=7.5, color=C_R)
a.annotate("", xy=(0.95, E_1A1), xytext=(0.45, E_ZPL + dz), arrowprops=dict(arrowstyle="-|>", color=C_S, lw=1.8, connectionstyle="arc3,rad=-0.15")); a.text(0.72, 1.52, "crossing to the singlets:\nstrong from m_s = ±1,\nweak from m_s = 0", fontsize=7.5, color=C_S, ha="center", va="top")
a.annotate("", xy=(1.1, E_1E), xytext=(1.1, E_1A1), arrowprops=dict(arrowstyle="-|>", color=C_S, lw=1.4)); a.text(1.12, (E_1E + E_1A1) / 2, "1042 nm", fontsize=7.5, color=C_S)
a.annotate("", xy=(0.45, 0), xytext=(0.95, E_1E), arrowprops=dict(arrowstyle="-|>", color=C_S, lw=1.8, ls="--", connectionstyle="arc3,rad=-0.15")); a.text(1.0, 0.2, "back mostly\nto m_s = 0", fontsize=7.5, color=C_S)
a.text(0.75, 2.8, "net effect: green light pumps the spin into m_s = 0 (bright);\na spin in m_s = ±1 takes the dark detour and fluoresces less",
       fontsize=7.8, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#bbbbbb"))
a.set(title="(b) the optical cycle: why the spin state is visible", xlim=(0, 1.5), ylim=(-0.15, 2.98), ylabel="energy above the ground state (eV)"); a.set_xticks([]); a.grid(False)
for sp in ("top", "right", "bottom"): a.spines[sp].set_visible(False)

# (c) the ground-state spin sublevels, three zooms
a = ax[2]
a.plot([0.05, 0.35], [0, 0], "k", lw=1.8); a.text(0.2, -0.09, "m_s = 0", ha="center", fontsize=8)
a.plot([0.05, 0.35], [1, 1], "k", lw=1.8); a.text(0.2, 1.08, "m_s = ±1", ha="center", fontsize=8)
a.annotate("", xy=(0.02, 1), xytext=(0.02, 0), arrowprops=dict(arrowstyle="<->", lw=0.8)); a.text(0.04, 0.5, "D = 2.870 GHz\n(zero field)", fontsize=7.5, va="center")
a.plot([0.35, 0.6], [1, 1.28], ":", color="gray", lw=0.8); a.plot([0.35, 0.6], [1, 0.72], ":", color="gray", lw=0.8)
a.plot([0.6, 0.9], [1.28, 1.28], "k", lw=1.8); a.text(0.75, 1.36, "m_s = +1", ha="center", fontsize=8)
a.plot([0.6, 0.9], [0.72, 0.72], "k", lw=1.8); a.text(0.75, 0.62, "m_s = −1", ha="center", fontsize=8)
a.plot([0.6, 0.9], [0, 0], "k", lw=1.8)
a.annotate("", xy=(0.68, 1.28), xytext=(0.68, 0), arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.4)); a.annotate("", xy=(0.82, 0.72), xytext=(0.82, 0), arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.4))
a.text(0.66, 0.42, "f₊ = D + γB∥", fontsize=7.5, color="C0", ha="right"); a.text(0.84, 0.3, "f₋ = D − γB∥", fontsize=7.5, color="C0")
a.annotate("", xy=(0.93, 1.28), xytext=(0.93, 0.72), arrowprops=dict(arrowstyle="<->", lw=0.8)); a.text(0.95, 1.0, "2γB∥ = 168 MHz\nat B∥ = 3 mT", fontsize=7.5, va="center")
a.plot([0.9, 1.15], [1.28, 1.56], ":", color="gray", lw=0.8); a.plot([0.9, 1.15], [1.28, 1.16], ":", color="gray", lw=0.8)
for k, lab in ((1, "m_I = +1"), (0, "m_I = 0"), (-1, "m_I = −1")):
    y = 1.36 + 0.14 * k; a.plot([1.15, 1.4], [y, y], "k", lw=1.4); a.text(1.42, y, lab, fontsize=7.5, va="center")
a.annotate("", xy=(1.12, 1.5), xytext=(1.12, 1.36), arrowprops=dict(arrowstyle="<->", lw=0.8)); a.text(1.02, 1.6, "hyperfine\n2.16 MHz", fontsize=7.5, ha="center")
a.text(0.75, 1.9, "microwave transitions the microscope reads:\nthe pair's splitting measures B∥, 28 MHz per mT;\none line's shift reads a change, 28 Hz per nT",
       fontsize=7.8, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#bbbbbb"))
a.set(title="(c) the ground-state spin levels, three zooms", xlim=(0, 1.62), ylim=(-0.2, 2.15)); a.set_xticks([]); a.set_yticks([]); a.grid(False)
for sp in ("top", "right", "bottom", "left"): a.spines[sp].set_visible(False)
a.text(0.2, -0.18, "GHz", ha="center", fontsize=7, color="gray"); a.text(0.75, -0.18, "MHz, zoomed", ha="center", fontsize=7, color="gray"); a.text(1.3, 0.82, "MHz, zoomed again", ha="center", fontsize=7, color="gray")

fig.suptitle("The nitrogen-vacancy centre: its levels in the band gap, the optical cycle that reads the spin, and the sublevels the microwave addresses", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig1_levels.png"); print("fig1")
