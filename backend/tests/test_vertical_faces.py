"""Uprights in a section. A utility pole standing in the graded section
changes nothing: the template runs on through it. A wall face or a curb is
part of the template, and the zone keeps its square corner at the face's
foot instead of a slope cut across it."""
import pytest

from gdot_earthwork.plan_set import process_plan_set

from .synth import curbed_street, cut_section, embankment, fill_wall, make_pdf, true_areas, walled_cut


def _measured(tmp_path, specs, scanned):
    pdf = tmp_path / "plans.pdf"
    make_pdf(str(pdf), [specs[i:i + 3] for i in range(0, len(specs), 3)], scanned=scanned)
    res = process_plan_set(str(pdf), workers=1, overlays=False)
    assert len(res.sections) == len(specs)
    # matched by position on the sheet (as drawn), not by the station read
    return [(s, specs[3 * (s.page - 1) + s.index]) for s in res.sections]


def _pole(spec, x):
    spec.poles, spec.pole_style = [x], "box"
    return spec


@pytest.mark.parametrize("scanned", [False, True], ids=["vector", "scan"])
def test_a_pole_in_the_graded_section_changes_nothing(tmp_path, scanned):
    specs = [_pole(embankment("21+00", 651.3, 5.0), 22.3),      # on the side slope
             _pole(embankment("20+50", 651.3, 5.0), -10.3),     # its guy passing under the roadway
             _pole(embankment("20+00", 651.3, 5.0), -25.3),     # guy anchored under the slope
             _pole(cut_section("19+50", 662.6, 5.0), -18.7),    # in the cut
             _pole(cut_section("19+00", 662.6, 5.0), 27.4)]     # on the back slope
    tol = 0.003 if not scanned else 0.01
    for s, spec in _measured(tmp_path, specs, scanned):
        fill, cut = true_areas(spec)
        assert abs(s.fill_ft2 - fill) + abs(s.cut_ft2 - cut) <= max(0.5, tol * (fill + cut)), \
            (spec.station, s.fill_ft2, fill, s.cut_ft2, cut)


@pytest.mark.parametrize("scanned", [False, True], ids=["vector", "scan"])
def test_walls_and_curbs_keep_their_square_corners(tmp_path, scanned):
    specs = [walled_cut("31+50", 660, 9.0), walled_cut("31+00", 660, 6.0), walled_cut("30+50", 660, 4.0),
             fill_wall("30+00", 651.3, 8.0), fill_wall("29+50", 651.3, 5.0), fill_wall("29+00", 651.3, 3.0),
             curbed_street("28+50", 662.3, 3.0, 0.67),    # curb tops and sidewalks on the 660 grid line
             curbed_street("28+00", 662.3, 3.0, 1.0), curbed_street("27+50", 662.3, 1.5, 0.5)]
    tol = 0.005 if not scanned else 0.025
    for s, spec in _measured(tmp_path, specs, scanned):
        fill, cut = true_areas(spec)
        assert abs(s.fill_ft2 - fill) + abs(s.cut_ft2 - cut) <= max(0.5, tol * (fill + cut)), \
            (spec.station, s.fill_ft2, fill, s.cut_ft2, cut)
