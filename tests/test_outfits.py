"""survsim.export.outfits: painted-clothing palettes for MetaHuman bodies."""
from survsim.export.outfits import BOTTOMS, TOPS, hivis_probability, outfit


def test_outfit_is_deterministic_and_varies():
    assert outfit("p1", 3) == outfit("p1", 3)
    looks = {(o["top"], o["bottom"], o["leg_cut"], o["sleeve"])
             for o in (outfit(f"p{i}", 0) for i in range(40))}
    assert len(looks) >= 15


def test_outfit_fields_are_in_range():
    names = {t[0] for t in TOPS} | {"hivis_yellow"}
    for i in range(50):
        o = outfit(f"x{i}", 1, hivis_p=0.3)
        assert o["top"] in names and o["bottom"] in {b[0] for b in BOTTOMS}
        assert 0.0 < o["hem_h"] < 0.4 and 0.2 < o["sleeve_u"] <= 0.8
        for k in ("top_rgb", "bottom_rgb", "shoes_rgb", "skin_rgb"):
            assert len(o[k]) == 3 and all(0.0 <= v <= 1.0 for v in o[k])


def test_street_wear_is_mostly_dark_and_hivis_only_on_work_sites():
    tops = [outfit(f"s{i}", 0)["top"] for i in range(400)]
    dark = sum(t in ("black", "navy", "charcoal", "olive", "burgundy") for t in tops)
    assert dark / len(tops) > 0.4
    assert "hivis_yellow" not in tops
    assert hivis_probability("perimeter_site") >= 0.2
    assert hivis_probability("event_plaza") < 0.05
