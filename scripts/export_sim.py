"""Export the data the interactive page needs: the two-loop scene, its field at the NV plane for
several stand-offs, and the constants of the photon budget. The browser does the rest (ODMR
frames, fitting, noise, and the Fourier inversion) live."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdm import current, photons

N, DX = 128, 0.5                      # 64 x 64 um field of view, 0.5 um pixels (a power of two for the browser FFT)
STANDOFFS = [0.5, 1.0, 2.0, 4.0]
x, y = current.grid(N, DX)
# the scene: a shallow loop in the top metal, a deeper loop in a lower metal; depths relative to the chip surface
LOOPS = [dict(name="M3 (top metal)", cx=-6, cy=-4, w=13, h=8, i_ua=20, depth_um=0.0, edge=0.5),
         dict(name="M1 (deep metal)", cx=7, cy=6, w=17, h=12, i_ua=60, depth_um=3.0, edge=0.5)]
maps = {}
jt = np.zeros((N, N))
for L in LOOPS:
    g = current.rounded_rect_stream(x, y, L["cx"], L["cy"], L["w"], L["h"], L["i_ua"] * 1e-6, L["edge"])
    jx, jy = current.currents_from_stream(g, DX)
    jt += np.hypot(jx, jy)
    for d in STANDOFFS:
        bz = current.bz_from_sheet(jx, jy, DX, d + L["depth_um"])
        maps[d] = maps.get(d, np.zeros((N, N))) + bz
out = dict(
    n=N, dx_um=DX, standoffs_um=STANDOFFS, loops=LOOPS,
    j_true=[[round(float(v), 2) for v in row] for row in jt],
    bz_uT={str(d): [[round(float(v) * 1e6, 4) for v in row] for row in maps[d]] for d in STANDOFFS},
    photons=dict(intensity_w_cm2=1e3, density_cm3=1e17, layer_um=1.0, pixel_um=DX,
                 collection=dict(bare=photons.collection_fraction(0.9), mirror=photons.collection_fraction(0.9, back_mirror=True), sil=photons.collection_fraction(0.9, sil=True)),
                 rate_per_collection=photons.photon_rate_per_pixel(1e3, 1e17, 1.0, DX, 1.0),
                 contrast=0.02, linewidth_hz=1e6, gamma_hz_per_t=28.024e9, k_cw=float(4 / (3 * np.sqrt(3)))),
    odmr=dict(d_ghz=2.870, bias_mT=2.0, axis_cos=float(np.cos(np.deg2rad(54.74)))),
)
dst = Path("/Users/emreacar/Documents/claude/Projects/Emre finding a job/reports/qdm-sim-data.json")
dst.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", dst, f"{dst.stat().st_size / 1e6:.2f} MB; peak Bz", {d: round(float(np.abs(m).max() * 1e6), 2) for d, m in maps.items()}, "uT")
