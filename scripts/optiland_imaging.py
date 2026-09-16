"""The imaging path in an optical design program: optiland (open-source sequential ray tracer).

The NV layer is imaged through the diamond by an ideal objective (a paraxial lens of focal
length f with the stop on it). Two exit surfaces are compared: the flat top face of a 300 um
plate, and a hemispherical solid immersion lens of 1 mm radius centred on the field. The
figure of merit is the RMS spot radius at the image, referred back to the NV layer by the
magnification: a plate at high NA is aberration-limited, the SIL is not.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from optiland import materials, optic

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
DIA = materials.IdealMaterial(n=2.417)
LAMBDA_UM = 0.7


def system(mode, obj_na, slab_mm=0.3, f_mm=2.0, sil_r_mm=1.0, wd_mm=3.0):
    o = optic.Optic()
    if mode == "flat":
        o.add_surface(index=0, radius=np.inf, thickness=slab_mm, material=DIA)     # NV layer, slab of diamond above it
        o.add_surface(index=1, radius=np.inf, thickness=wd_mm, material="air")     # flat top face, working distance
    else:
        o.add_surface(index=0, radius=np.inf, thickness=sil_r_mm, material=DIA)    # NV layer at the centre of the hemisphere
        o.add_surface(index=1, radius=-sil_r_mm, thickness=wd_mm, material="air")  # spherical exit face
    o.add_surface(index=2, surface_type="paraxial", f=f_mm, thickness=f_mm, is_stop=True, material="air")
    o.add_surface(index=3)
    o.set_aperture(aperture_type="objectNA", value=obj_na)
    o.set_field_type("object_height")
    for y in (0.0, 0.016, 0.032):
        o.add_field(y=y)
    o.add_wavelength(value=LAMBDA_UM, is_primary=True)
    o.image_solve()
    return o


def spot(o, hy, num_rays=64):
    r = o.trace(Hx=0, Hy=hy, wavelength=LAMBDA_UM, num_rays=num_rays, distribution="hexapolar")
    x, y = np.asarray(r.x).ravel(), np.asarray(r.y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    m = abs(float(np.asarray(o.paraxial.magnification()).ravel()[0]))
    x, y = (x[ok] - x[ok].mean()) / m * 1e3, (y[ok] - y[ok].mean()) / m * 1e3     # um, referred to the NV layer
    return x, y, float(np.sqrt(np.mean(x ** 2 + y ** 2)))


if __name__ == "__main__":
    cases = [("flat", 0.37, "flat 300 µm plate, air NA 0.89"), ("flat", 0.20, "flat plate, air NA 0.48"), ("sil", 0.9, "SIL R = 1 mm, NA 0.9")]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
    rows = []
    airy = 0.61 * LAMBDA_UM / 0.9
    for a, (mode, na, lab) in zip(ax, cases):
        o = system(mode, na); pair = []
        for hy, fy, col in ((0.0, 0, "C0"), (1.0, 32, "C3")):
            x, y, rms = spot(o, hy)
            a.plot(x, y, ".", ms=2, color=col, alpha=.6, label="centre of the field" if fy == 0 else "32 µm off centre")
            rows.append((lab, fy, rms)); pair.append(rms)
        c = plt.Circle((0, 0), airy, fill=False, color="k", ls=":", lw=1); a.add_patch(c)
        a.set(xlabel="x at the NV layer (µm)", ylabel="y (µm)", xlim=(-3, 3), ylim=(-3, 3)); a.set_title(f"{lab}\nRMS {pair[0]:.2f} µm at the centre, {pair[1]:.2f} µm at 32 µm off", fontsize=10); a.set_aspect("equal"); a.grid(alpha=.3)
    ax[1].legend(fontsize=8, loc="upper right"); ax[1].text(-2.9, -2.8, f"dotted: Airy radius {airy:.2f} µm at NA 0.9", fontsize=8)
    fig.suptitle("Spot diagrams at the NV layer through an ideal objective (optiland 0.6.1): the diamond plate adds spherical aberration, the SIL does not", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "fig12_optiland.png"); print("fig12")
    for lab, fy, rms in rows:
        print(f"{lab:32s} field {fy:2d} µm  RMS spot {rms:.3f} µm")
