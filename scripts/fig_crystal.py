"""Figure 2: the diamond lattice behind the four NV orientations, and what a (100) or a (111) plate does
to the angles between the NV axes and the surface normal."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
fig = plt.figure(figsize=(12.5, 4.9))

# (a) the conventional cubic cell
a = fig.add_subplot(1, 3, 1, projection="3d")
fcc = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)] + [(.5, .5, 0), (.5, .5, 1), (.5, 0, .5), (.5, 1, .5), (0, .5, .5), (1, .5, .5)]
inner = [(.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)]
for p in inner:
    for q in fcc:
        if abs(np.linalg.norm(np.subtract(p, q)) - np.sqrt(3) / 4) < 1e-6:
            a.plot(*zip(p, q), color="#8a8a8a", lw=1.3)
for s, e in [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (1, 1, 0)),
             ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]:
    a.plot(*zip(s, e), color="#c8c8c8", lw=0.8)
pts = [p for p in fcc if p != (0, 0, 0)] + inner[1:]
a.scatter(*zip(*pts), s=38, color="#4a4a4a", depthshade=False)
a.scatter([.25], [.25], [.25], s=90, color="#2f6fd1", depthshade=False)                      # the nitrogen
a.scatter([0], [0], [0], s=110, facecolors="none", edgecolors="k", linestyle="--", depthshade=False)   # the vacancy
a.quiver(0, 0, 0, .55, .55, .55, color="#d1493a", lw=2, arrow_length_ratio=.15)
a.text(.27, .2, .32, "N", color="#2f6fd1", fontsize=9, fontweight="bold"); a.text(-.05, -.12, .02, "V", fontsize=9, fontweight="bold")
a.text2D(0.64, 0.1, "N–V axis along [111]", color="#d1493a", fontsize=8.5, transform=a.transAxes)
a.text(1.02, -.05, -.08, "[100]", fontsize=8); a.text(-.12, -.05, 1.0, "[001]", fontsize=8)
a.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1)); a.set_box_aspect((1, 1, 1)); a.set_axis_off(); a.view_init(elev=20, azim=-40)
a.set_title("(a) the diamond cubic cell: bonds along ⟨111⟩,\nand the NV axis is one of them", fontsize=9.5)

def plate(ax, axes, normal_label, angle_text, angle_pos, title):
    X, Y = np.meshgrid([-1, 1], [-1, 1])
    for z0, alpha in ((0.0, .35), (-.22, .2)):
        ax.plot_surface(X, Y, np.full_like(X, z0), color="#9fd3ff", alpha=alpha, edgecolor="#6aa8d8", lw=.5)
    for i, v in enumerate(axes):
        v = np.array(v, float); v /= np.linalg.norm(v)
        if v[2] < 0: v = -v
        ax.quiver(0, 0, 0, *(0.95 * v), color="#d1493a", lw=1.8, arrow_length_ratio=.12)
    ax.quiver(0, 0, 0, 0, 0, 1.0, color="k", lw=1.2, arrow_length_ratio=.1, linestyle="--")
    ax.text(0.03, 0.03, 1.02, normal_label, fontsize=8); ax.text2D(angle_pos[0], angle_pos[1], angle_text, fontsize=9, color="#d1493a", transform=ax.transAxes)
    ax.text(-1.0, -1.0, -0.5, "polished plate", fontsize=8, color="#3b7fb5")
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-.4, 1.1)); ax.set_box_aspect((1, 1, .75)); ax.set_axis_off(); ax.view_init(elev=18, azim=-55); ax.set_title(title, fontsize=9.5)

axes100 = [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]
plate(fig.add_subplot(1, 3, 2, projection="3d"), axes100, "[001] normal", "54.7° from the normal,\nall four axes", (0.56, 0.8), "(b) a (100) plate: the normal is a cube axis,\nevery NV axis makes 54.7° with it")
# (111) plate: rotate so that [111] becomes the normal
R = np.array([[1, -1, 0] / np.sqrt(2), [1, 1, -2] / np.sqrt(6), [1, 1, 1] / np.sqrt(3)])
axes111 = [R @ np.array(v, float) for v in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
plate(fig.add_subplot(1, 3, 3, projection="3d"), axes111, "[111] normal", "one axis along the normal,\nthree at 70.5°", (0.56, 0.8), "(c) a (111) plate: one NV axis is the normal,\nthe other three lie 70.5° from it")
fig.suptitle("The crystal behind the four orientations: bonds along ⟨111⟩, and the angle each NV axis makes with the plate's surface", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig2_crystal.png"); print("fig2")
