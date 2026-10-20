"""
End-to-end checks on synthetic sheets whose areas are known exactly:
the vector path, the scanned (raster) path, road grouping and volumes.
"""
import pytest

from gdot_earthwork import process_plan_set
from gdot_earthwork.volume import average_end_area_volume

from .synth import cut_section, embankment, make_pdf, true_areas

SPECS = [
    [embankment("12+00", 651.3, 6.0), embankment("11+50", 652.1, 5.0, tilt=0.02), embankment("11+00", 651.7, 4.0)],
    [cut_section("13+50", 663.4, 5.0), cut_section("13+00", 662.6, 4.0), embankment("12+50", 651.2, 7.0)],
]
BY_STATION = {s.station: s for page in SPECS for s in page}


@pytest.fixture(scope="module")
def vector_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("vec") / "plans.pdf"
    make_pdf(str(path), SPECS)
    return process_plan_set(str(path), workers=1, overlays=False)


@pytest.fixture(scope="module")
def scanned_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("scan") / "plans.pdf"
    make_pdf(str(path), SPECS, scanned=True)
    return process_plan_set(str(path), workers=1, overlays=False)


def _check_areas(result, tol_rel, tol_abs):
    assert len(result.sections) == 6
    for s in result.sections:
        fill, cut = true_areas(BY_STATION[s.station_label])
        assert s.fill_ft2 == pytest.approx(fill, rel=tol_rel, abs=tol_abs), s.station_label
        assert s.cut_ft2 == pytest.approx(cut, rel=tol_rel, abs=tol_abs), s.station_label


def test_vector_sections_found_in_station_order(vector_result):
    assert [s.station_label for s in vector_result.sections] == \
        ["11+00", "11+50", "12+00", "12+50", "13+00", "13+50"]
    assert all(s.method == "vector" and s.label_source == "text" for s in vector_result.sections)


def test_vector_areas_match_exact_geometry(vector_result):
    _check_areas(vector_result, tol_rel=0.002, tol_abs=0.3)   # read off the drawn geometry


def test_scanned_sheets_read_and_measured(scanned_result):
    assert all(s.method == "raster" for s in scanned_result.sections)
    labels = [s.station_label for s in scanned_result.sections]
    assert labels == ["11+00", "11+50", "12+00", "12+50", "13+00", "13+50"]
    _check_areas(scanned_result, tol_rel=0.04, tol_abs=5.0)


def test_one_road_and_volume(vector_result):
    assert len(vector_result.roads) == 1
    vol = vector_result.volumes[0]
    assert len(vol.segments) == 5
    expected = average_end_area_volume(
        [type("S", (), dict(station_label=k, station_ft=float(k.replace("+", "")),
                            fill_ft2=true_areas(v)[0], cut_ft2=true_areas(v)[1]))() for k, v in sorted(BY_STATION.items())])
    assert vol.total_fill_cy == pytest.approx(expected.total_fill_cy, rel=0.02)
    assert vol.total_cut_cy == pytest.approx(expected.total_cut_cy, rel=0.02)
