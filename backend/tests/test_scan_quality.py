"""Real scans, not clean renderings: a sheet fed in at an angle, soft
optics, paper grain, dust and JPEG compression, at the resolutions plan sets
are scanned at. The grid must still be found and erased, the axis ticks
kept out of the template, and every section measured."""
import cv2
import numpy as np
import pytest

from gdot_earthwork.pages import skew_angle
from gdot_earthwork.plan_set import process_plan_set

from .synth import Scan, cut_section, embankment, make_pdf, true_areas


@pytest.mark.parametrize("tilt", [0.0, 0.05, 0.3, -0.8, 2.2])
def test_tilt_is_measured(tilt):
    g = np.full((2800, 4300), 255, np.uint8)
    for y in range(200, 2600, 127):
        cv2.line(g, (100, y), (4200, y), 0, 2)
    for x in range(100, 4200, 127):
        cv2.line(g, (x, 200), (x, 2600), 0, 2)
    g = cv2.warpAffine(g, cv2.getRotationMatrix2D((2150, 1400), tilt, 1.0), (4300, 2800), borderValue=255)
    assert skew_angle(g) == pytest.approx(tilt, abs=0.01)


def test_a_sheet_without_long_lines_is_left_level():
    g = np.full((2800, 4300), 255, np.uint8)
    cv2.putText(g, "GENERAL NOTES", (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, 0, 8)
    assert skew_angle(g) == 0.0


SPECS = [cut_section("10+50", 662.6, 4.0),
         cut_section("10+00", 662.6, 2.5),           # roadbed on the 660 grid line
         embankment("9+50", 651.3, 2.5)]


@pytest.mark.parametrize("dpi, scan", [
    (254, Scan()),                                         # compressed: the axis ticks join up
    (254, Scan(tilt=-0.8)),
    (200, Scan(tilt=0.3, blur=0.8, noise=12)),
    (150, Scan(tilt=-0.5)),
], ids=["jpeg", "tilted", "200dpi-soft-noisy", "150dpi-tilted"])
def test_degraded_scans_are_measured(tmp_path, dpi, scan):
    pdf = tmp_path / "scan.pdf"
    make_pdf(str(pdf), [SPECS], dpi=dpi, scan=scan)
    res = process_plan_set(str(pdf), workers=1, overlays=False)
    assert len(res.sections) == 3
    for s in res.sections:
        spec = SPECS[s.index]
        fill, cut = true_areas(spec)
        assert abs(s.fill_ft2 - fill) + abs(s.cut_ft2 - cut) <= max(3.0, 0.03 * (fill + cut)), \
            (spec.station, s.fill_ft2, fill, s.cut_ft2, cut)


def test_shapes_land_on_the_original_tilted_scan(tmp_path):
    """Sections are read off the straightened scan; what is drawn back onto
    the PDF (frames, traced lines, cut / fill zones) must land where the
    drawing is on the tilted original."""
    import pymupdf as fitz

    pdf = tmp_path / "tilted.pdf"
    make_pdf(str(pdf), [SPECS[:1]], scan=Scan(tilt=0.8, jpeg=95, dust=0))
    clean = tmp_path / "level.pdf"
    make_pdf(str(clean), [SPECS[:1]])
    res = process_plan_set(str(pdf), workers=1, overlays=False)
    level = process_plan_set(str(clean), workers=1, overlays=False)
    tl, tr, bl = res.sections[0].geometry["frame"]
    # the same corners on the level sheet, turned 0.8 deg about the page centre as the scan was
    w, h = fitz.open(str(clean))[0].rect.width, fitz.open(str(clean))[0].rect.height
    a = np.radians(0.8)
    turn = lambda p: ((p[0] - w / 2) * np.cos(a) + (p[1] - h / 2) * np.sin(a) + w / 2,
                      -(p[0] - w / 2) * np.sin(a) + (p[1] - h / 2) * np.cos(a) + h / 2)
    for got, want in zip((tl, tr, bl), level.sections[0].geometry["frame"]):
        assert np.hypot(*np.subtract(got, turn(want))) < 1.0, (got, turn(want))     # PDF points
