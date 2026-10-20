"""Station labels on scans: read glyph by glyph (digits.py) and repaired from
the station sequence (stations.py). A wrong station puts a section's area on
the wrong stretch of road, so every misread changes the volumes."""
import cv2
import numpy as np
import pymupdf as fitz
import pytest

from gdot_earthwork.digits import read_station
from gdot_earthwork.plan_set import process_plan_set
from gdot_earthwork.stations import StationRead, _single_edits, format_station, repair

from .synth import Scan, cut_section, embankment, make_pdf


def _word(text: str, font: str, px: int, tilt: float = 0.0) -> np.ndarray:
    """A label lettered in `font`, its digits `px` tall, scanned at a tilt and
    turned level again (as pages.straighten does)."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=60)
    page.insert_text((10, 40), text, fontsize=24, fontname=font)
    pix = page.get_pixmap(dpi=int(72 * px / 17), colorspace=fitz.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(np.float32)
    for a in ((tilt, -tilt) if tilt else ()):
        g = cv2.warpAffine(g, cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), a, 1.0),
                           (g.shape[1], g.shape[0]), borderValue=255)
    return np.where(g < 200, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("font", ["helv", "tiro", "cour", "hebo"])
@pytest.mark.parametrize("tilt", [0.0, 0.8])
def test_labels_are_read_glyph_by_glyph(font, tilt):
    for text in ("106+00", "11+00", "9+50", "213+25", "1047+75", "48+05"):
        got = read_station(_word(text, font, 22, tilt))
        assert got and got[0] == text, (font, tilt, text, got)


def test_a_word_not_shaped_like_a_station_is_not_read():
    assert read_station(_word("670", "helv", 22)) is None            # an elevation label
    assert read_station(_word("12+5", "helv", 22)) is None


def test_single_digit_slips_are_offered_as_corrections():
    assert "108+00" in _single_edits("198+00")                         # a misread middle digit
    assert "39+00" in _single_edits("38+00")
    assert "213+50" in _single_edits("213+30")                         # after the plus, to a 25 ft station
    assert "102+54" not in _single_edits("102+50")                     # never made up off the 25 ft stations


def test_a_misread_digit_at_the_end_of_a_road_is_corrected():
    reads = [StationRead(format_station(3100 + 50 * i), 3100 + 50 * i, 1.0, "ocr") for i in range(16)]   # to 38+50
    reads += [StationRead("38+00", 3800.0, 0.6, "ocr"), StationRead("38+50", 3850.0, 0.6, "ocr")]   # 39+00, 39+50
    assert [r.label for r in repair(reads)][-2:] == ["39+00", "39+50"]


def test_stations_lettered_over_the_grid_on_a_noisy_scan(tmp_path):
    """Three-digit stations, lettered across a grid line as hand-drafted and
    some CAD sheets are, on a soft, noisy, tilted 200 dpi scan: blanking the
    grid splits the "0" of "104" and the "1" beside it reads as a "9"."""
    stations = [format_station(10400 + 25 * i) for i in range(6)]
    specs = [cut_section(s, 662.6, 3.0) if i % 2 else embankment(s, 651.3, 3.0) for i, s in enumerate(stations)]
    pdf = tmp_path / "scan.pdf"
    make_pdf(str(pdf), [specs[:3], specs[3:]], dpi=200, scan=Scan(tilt=0.3, blur=0.8, noise=12))
    res = process_plan_set(str(pdf), workers=1, overlays=False)
    assert [s.station_label for s in res.sections] == stations
