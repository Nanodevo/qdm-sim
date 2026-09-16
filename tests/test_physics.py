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
