"""Drawn lines that meet the sheet's grid: a roadbed lying on a grid line,
and ground passing close by the template where both cross one. Removing the
grid must cost neither line any area."""
import cv2
import numpy as np
import pytest

from gdot_earthwork.plan_set import process_plan_set
from gdot_earthwork.tracing import strip_grid

from .synth import curbed_street, cut_section, make_pdf, true_areas


@pytest.mark.parametrize("scanned", [False, True], ids=["vector", "scan"])
def test_roadbed_on_a_grid_line_is_measured_whole(tmp_path, scanned):
    # base of the template at 660.1 and 659.9: 24 ft of roadbed on the 660
    # grid line, wider than the tracing window once the grid is erased
    specs = [cut_section("21+00", 662.6, 2.5), cut_section("20+50", 662.6, 2.7)]
    pdf = tmp_path / "grid.pdf"
    make_pdf(str(pdf), [specs], scanned=scanned)
    res = process_plan_set(str(pdf), workers=1, overlays=False)
    assert len(res.sections) == 2
    for s, spec in zip(res.sections, specs[::-1]):
        cut = true_areas(spec)[1]
        assert abs(s.cut_ft2 - cut) <= 0.02 * cut, (s.station_label, s.cut_ft2, cut)


def test_sidewalks_on_a_grid_line_stay_with_the_street(tmp_path):
    # the bottom street's curb tops and sidewalks lie on the 660 grid line:
    # erased with it, the side slopes behind them are left as short strokes
    # 6 ft from the rest of the template
    specs = [curbed_street("15+50", 662.3, 3.0, 0.5), curbed_street("15+00", 662.3, 1.5, 0.67),
             curbed_street("14+50", 662.3, 3.0, 0.67)]
    pdf = tmp_path / "curbs.pdf"
    make_pdf(str(pdf), [specs], scanned=True)
    res = process_plan_set(str(pdf), workers=1, overlays=False)
    assert len(res.sections) == 3
    for s in res.sections:
        spec = specs[s.index]
        cut = true_areas(spec)[1]
        assert abs(s.cut_ft2 - cut) <= 0.02 * cut, (spec.station, s.cut_ft2, cut)


def test_ground_near_the_template_is_not_welded_to_it_across_a_grid_line():
    ink = np.zeros((60, 40), np.uint8)
    ink[20, :] = 1                   # ground
    ink[28, :] = 1                   # template, 8 px below
    ink[:, 18:21] = 1                # a vertical grid line crossing both
    m = strip_grid(ink, [(18, 20)], [])
    assert m[20, 18:21].all() and m[28, 18:21].all()      # both lines carried across
    assert not m[21:28, 18:21].any()                      # nothing in between


def test_a_thin_shallow_line_stays_whole_across_a_heavier_grid_line():
    ink = np.zeros((40, 80), np.uint8)
    ink[10:15, :] = 1                # a heavy grid row
    ink[9, :30] = 1                  # a thin line running in above it ...
    ink[15, 33:] = 1                 # ... and out below, 3 columns on
    m = strip_grid(ink, [], [(10, 14)])
    n, _ = cv2.connectedComponents(m, connectivity=8)
    assert n == 2                    # background + one line

    assert not m[10:15, :20].any() and not m[10:15, 45:].any()   # the grid row itself stays erased
