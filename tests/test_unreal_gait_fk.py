"""FK on the CesiumMan walk clip: calibrates play rate and the stance pose."""
from pathlib import Path

import pytest

from survsim.export.gait_fk import gait_stats

GLB = Path.home() / "data/glb/CesiumMan.glb"


@pytest.mark.skipif(not GLB.is_file(), reason="CesiumMan.glb not present")
def test_cesiumman_gait_is_plausible():
    g = gait_stats(GLB, "leg_joint_L_5", "leg_joint_R_5", up_axis=2)
    assert g["cycle_s"] == pytest.approx(2.0)
    # A ~1.5 m figure: step 0.5-0.9 m, ground speed 0.5-1.2 m/s.
    assert 0.5 < g["step_length"] < 0.9
    assert 0.5 < g["ground_speed"] < 1.2
    assert abs(g["foot_min_up"]) < 0.05          # feet reach the ground
    assert 0.0 <= g["stance_t"] < g["cycle_s"]
