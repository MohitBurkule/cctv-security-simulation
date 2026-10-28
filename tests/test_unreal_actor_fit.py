from survsim.export.actor_fit import fit, violations


def test_giant_is_scaled_and_grounded():
    f = fit(origin=(0, 0, 900), extent=(100, 60, 1000), ground_z=0, target_height_cm=175)
    assert f["upright_axis"] == "z"
    assert abs(f["scale"] - 175 / 2000) < 1e-9
    assert f["dz"] == 100.0


def test_lying_actor_is_flagged():
    v = violations(origin=(0, 0, 20), extent=(90, 30, 20), ground_z=0)
    assert any("not upright" in x for x in v)


def test_good_actor_passes():
    assert violations(origin=(0, 0, 87.5), extent=(30, 25, 87.5), ground_z=0) == []


def test_sunk_actor_is_flagged():
    assert any("ground" in x for x in violations((0, 0, 60), (30, 25, 87.5), 0))
