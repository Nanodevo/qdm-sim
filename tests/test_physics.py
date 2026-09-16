import numpy as np

from qdm import current, nv, optics, photons


def test_zeeman_splitting_along_axis():
    ax = nv.NV_AXES[0]
    b = 5e-3 * ax                     # 5 mT along the axis
    f_minus, f_plus = nv.transitions_ghz(b, ax)
    assert abs((f_plus - f_minus) - 2 * nv.GAMMA_GHZ_PER_T * 5e-3) < 1e-6
    assert abs((f_plus + f_minus) / 2 - nv.D_GHZ) < 1e-6
    assert abs(nv.field_from_axis_splitting(f_plus, f_minus) - 5e-3) < 1e-9


def test_eight_lines_for_generic_field():
    b = 3e-3 * np.array([0.3, 0.5, 0.81])
    lines = nv.all_transitions_ghz(b).ravel()
    assert len(np.unique(np.round(lines, 5))) == 8


def test_collection_numbers():
    assert abs(photons.escape_cone_fraction() - 0.0455) < 0.002
    assert photons.collection_fraction(0.9, sil=True) > 5 * photons.collection_fraction(0.9)
    eta1 = photons.sensitivity_t_per_rthz(1e6)
    eta4 = photons.sensitivity_t_per_rthz(4e6)
    assert abs(eta1 / eta4 - 2.0) < 1e-9


def test_wire_peaks_at_standoff():
    x = np.linspace(-10, 10, 4001)
    b = optics.wire_bz(x, 1e-3, 2.0)
    assert abs(x[np.argmax(b)] - 2.0) < 0.02


def test_fourier_forward_matches_biot_savart_and_inverts():
    n, dx = 128, 0.5
    x, y = current.grid(n, dx)
    g = current.rounded_rect_stream(x, y, 0, 0, 20, 12, 5e-3, 1.0)
    jx, jy = current.currents_from_stream(g, dx)
    b_f = current.bz_from_sheet(jx, jy, dx, 1.5)
    b_d = current.bz_biot_savart(jx, jy, dx, 1.5)
    assert np.corrcoef(b_f.ravel(), b_d.ravel())[0, 1] > 0.999
    assert abs(b_f.max() / b_d.max() - 1) < 0.05
    jxr, jyr = current.reconstruct_currents(b_f, dx, 1.5, k_cut_per_um=4.0)
    assert current.rms_error((jxr, jyr), (jx, jy)) < 0.25


def test_material_directions():
    from qdm import material
    assert material.nv_density_cm3(10, 0.1) < material.nv_density_cm3(10, 1.0) < material.nv_density_cm3(10, 10.0)
    assert material.nv_density_cm3(10, 10.0) / material.nv_density_cm3(10, 100.0) > 0.99   # saturates
    assert material.t2_star_s(1.0) > material.t2_star_s(10.0) > material.t2_star_s(100.0)
    assert material.t2_star_s(1.0, c13_fraction=1e-4) > material.t2_star_s(1.0)              # 12C enrichment helps
    assert material.active_layer_um(0.1, "hydrogen") < material.active_layer_um(0.1, "oxygen")


def test_protocol_relations():
    from qdm import protocols
    s = np.linspace(0.2, 10, 500)
    eta = np.array([protocols.eta_cw(1e6, 1e6, 0.02, v) for v in s])
    assert abs(s[np.argmin(eta)] - 2.0) < 0.1                       # CW optimum at s = 2
    assert protocols.eta_ramsey(1e6, 10e-6, 0.02) < protocols.eta_ramsey(1e6, 1e-6, 0.02)   # longer T2* helps
    # pulsed ODMR pays a readout duty cycle; it beats CW when the readout window is a fair share of the cycle
    assert protocols.eta_pulsed_odmr(1e6, 10e-6, 0.02, t_read_s=5e-6, t_init_s=1e-6) < protocols.eta_cw(1e6, 1 / (np.pi * 10e-6), 0.02)
    assert protocols.eta_pulsed_odmr(1e6, 10e-6, 0.02, t_read_s=0.3e-6, t_init_s=2e-6) > protocols.eta_cw(1e6, 1 / (np.pi * 10e-6), 0.02)


def test_two_layer_inversion_separates_known_depths():
    n, dx = 128, 0.5
    x, y = current.grid(n, dx)
    g1 = current.rounded_rect_stream(x, y, -6, -4, 13, 8, 20e-6, 0.5)
    g2 = current.rounded_rect_stream(x, y, 7, 6, 17, 12, 60e-6, 0.5)
    bz = current.bz_from_sheet(*current.currents_from_stream(g1, dx), dx, 1.0) + current.bz_from_sheet(*current.currents_from_stream(g2, dx), dx, 4.0)
    rng = np.random.default_rng(1)
    bz_n = bz + rng.normal(0, 0.22e-6, bz.shape)
    masks = (current.layout_mask(g1, dx), current.layout_mask(g2, dx))
    (_, _), (_, _), (r1, r2) = current.reconstruct_two_layers(bz_n, dx, 1.0, 4.0, snr=15, masks=masks)
    in1 = g1 > 0.5 * g1.max(); in2 = g2 > 0.5 * g2.max()
    i1, i2 = current.loop_current_a(r1, in1), current.loop_current_a(r2, in2)
    assert abs(i1 - 20e-6) < 6e-6 and abs(i2 - 60e-6) < 15e-6


def test_vertical_segment_has_no_bz_and_vector_recovers_projections():
    from qdm import segments
    X, Y = np.meshgrid(np.linspace(-5e-6, 5e-6, 41), np.linspace(-5e-6, 5e-6, 41))
    bx, by, bz = segments.segment_field((0, 0, -4e-6), (0, 0, -1e-6), 1e-3, X, Y, 0.0)
    assert np.abs(bz).max() < 1e-12 * np.abs(bx).max() + 1e-18
    assert np.abs(bx).max() > 1e-6                              # the in-plane field is there
    p = segments.projections(bx, by, bz)
    rx, ry, rz = segments.vector_from_projections(p)
    assert np.allclose(rx, bx) and np.allclose(ry, by) and np.allclose(rz, bz, atol=1e-15)


def test_long_wire_limit():
    from qdm import segments
    X, Y = np.meshgrid(np.array([0.0]), np.array([2e-6]))
    bx, by, bz = segments.segment_field((-1.0, 0, 0), (1.0, 0, 0), 1e-3, X, Y, 0.0)   # 2 m wire, 2 um away
    assert abs(bz[0, 0] - 4e-7 * np.pi * 1e-3 / (2 * np.pi * 2e-6)) / (4e-7 * np.pi * 1e-3 / (2 * np.pi * 2e-6)) < 1e-6


def test_bz_determines_the_inplane_field_even_with_vias():
    """Above all sources Bz fixes Bx and By (potential field): the residual of the Hilbert
    prediction is a finite-window artifact that falls with window size, vias or not."""
    from qdm import segments
    u = 1e-6
    path = [(-60*u,-4*u,-1*u),(-6*u,-4*u,-1*u),(-6*u,-4*u,-4*u),(14*u,-4*u,-4*u),(14*u,-4*u,-1*u),(60*u,-4*u,-1*u),(60*u,60*u,-1*u),(-60*u,60*u,-1*u),(-60*u,-4*u,-1*u)]
    res = []
    for n in (128, 256):
        ax = (np.arange(n) - n // 2) * 0.5e-6
        X, Y = np.meshgrid(ax, ax)
        bx, by, bz = segments.path_field(path, 50e-6, X, Y, 0.0)
        r = segments.hilbert_residual(bx, by, bz, 0.5e-6)
        res.append(r[np.argmin(abs(ax + 4e-6)), np.argmin(abs(ax + 6e-6))] / np.hypot(bx, by).max())   # at the via
    assert res[0] < 0.05 and res[1] < 0.01 and res[1] < res[0] / 3


def test_ray_traced_collection_agrees_with_solid_angle_estimates():
    from qdm import photons, rays
    # bare flat surface: the Monte Carlo integrates the angle-dependent Fresnel loss the estimate averages
    mc = rays.collection_mc(0.9, "bare", n_rays=200_000)
    est = photons.collection_fraction(0.9)
    assert abs(mc - est) / est < 0.2
    # solid immersion lens, uncoated: the estimate assumed no surface loss, the trace pays ~17 %
    mc_sil = rays.collection_mc(0.9, "sil", n_rays=200_000)
    est_sil = photons.collection_fraction(0.9, sil=True)
    assert 0.7 * est_sil < mc_sil < est_sil
    assert abs(rays.collection_mc(0.9, "sil", n_rays=200_000, sil_coated=True) - est_sil) / est_sil < 0.05
    # a mirrored back side roughly doubles the bare case
    assert 1.7 < rays.collection_mc(0.9, "mirror", n_rays=200_000) / mc < 2.1


def test_illumination_refraction():
    from qdm import rays
    f = rays.illumination_footprint(53.6, 36.0, 300.0)
    assert 19 < f["theta_t_deg"] < 20 and 100 < f["shift_um"] < 112 and 0.78 < f["fresnel_t"] < 0.84
    assert abs(f["semi_axes_um"][1] / f["semi_axes_um"][0] - 1.68) < 0.02
