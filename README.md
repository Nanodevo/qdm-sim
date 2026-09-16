# qdm-sim: a system model of a widefield quantum diamond microscope

A quantum diamond microscope images the magnetic field of the currents inside a
chip with a layer of nitrogen-vacancy centres in a diamond placed on it. The
field shifts the NV spin resonances; a camera reads those shifts pixel by pixel
under microwave and green-laser illumination; an inversion turns the field map
back into a current map. This repository models that chain end to end, from the
spin Hamiltonian to the reconstructed current, so that questions like "what does
a better collection optic buy in field sensitivity" or "what limits the spatial
resolution" have numbers.

```
qdm.nv        spin-1 Hamiltonian, four <111> orientations, ODMR lines and spectra
qdm.photons   escape cone, collection fraction (bare, back mirror, SIL), photons per pixel,
              shot-noise-limited sensitivity (Dreau et al. 2011)
qdm.material  nitrogen content, irradiation dose, isotopes, strain and surface termination
              -> NV density, T2*, linewidth, active layer, contrast (Bauch 2020 scalings)
qdm.protocols CW (power-broadened), pulsed ODMR and Ramsey sensitivities with duty cycle (Barry 2020)
qdm.optics    diffraction, wire response, the stand-off as a low-pass filter
qdm.current   sheet currents -> Bz at the NV plane (Fourier Biot-Savart, Roth 1989)
              -> current map back from Bz with a windowed inversion; two layers at known
              depths (spectral split, layout-constrained alternation, template fit of net currents)
qdm.rays      Monte Carlo ray tracing of the collection (flat face, mirrored back, solid immersion
              lens, Fresnel and TIR) and of the illumination beam's refraction into the diamond
qdm.segments  3D current paths with vias (closed-form Biot-Savart per segment), the field vector
              as the four NV orientations measure it, net-current fit to the vector map,
              Bz recovered from a single orientation's projection map
scripts/fig_levels.py figure 1 (level diagram); fig_crystal.py 2 (lattice, plate orientations); figs.py 3, 4, 7, 8;
                  figs_material.py 5, 6; fig_two_layer.py 9; fig_vector.py 10; fig_rays.py 11;
                  optiland_imaging.py 12 (the imaging path in optiland, an open-source sequential ray tracer);
                  export_sim.py the data of the interactive page
tests/            14 checks: Zeeman splitting, eight lines, collection numbers (estimates and ray trace),
                  wire peaks, Fourier forward against direct Biot-Savart, forward-inverse round trip,
                  material and protocol relations, two-layer and vector recovery, Bz fixing the
                  in-plane field above all sources, a single-orientation projection converted to Bz,
                  the beam's refraction into the diamond
```

Study page: https://nanodevo.github.io/reports/qdm-sim.html

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
python -m pytest -q
python scripts/figs.py
```

Assumptions, stated once: isotropic NV emission (the average over the four
orientations), photon shot noise as the only noise,
Lorentzian lines of fixed contrast and width, sheet currents at a single depth
for the basic inversion (qdm.current lifts that with known depths and layout,
qdm.segments with three-dimensional paths). The collection fractions of
qdm.photons are solid-angle and Fresnel estimates, checked by the Monte Carlo
trace in qdm.rays; the imaging path is modelled in optiland with a paraxial
objective (scripts/optiland_imaging.py), so the objective's own surfaces and
tolerances stay open. Every number in the figures follows from those
assumptions and the constants at the top of each module.
