"""
Synthetic cross-section sheets with exactly known answers.

Draws GDOT-style sections -- 10 ft grid, offset axis with ticks and
labels, station label, dashed existing ground, solid proposed template --
into a vector PDF, and optionally a "scanned" copy (the same page rasterized
and embedded as a single image, with no text layer). Because the geometry
is generated, the true fill/cut area of every section is known exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pymupdf as fitz

PT_PER_FT = 3.6          # drawing scale: 1 in = 20 ft  ->  72 pt / 20 ft
GRID_FT = 10.0


@dataclass
class SectionSpec:
    station: str                         # e.g. "12+50"
    ground: list[tuple[float, float]]    # (offset ft, elevation ft) polyline -- the TOP of what exists
    design: list[tuple[float, float]]
    pavement: list[tuple[float, float, float]] = field(default_factory=list)  # existing asphalt: (from, to, thick ft)
    poles: list[float] = field(default_factory=list)                          # utility poles standing on the ground
    pole_style: str = "line"     # "line": an upright with a cross arm; "box": a 1 ft box carried 6 ft into the
                                 # ground, with a cross arm and a down guy anchored 12 ft to the right
    stage_mark: tuple[str, float, float] | None = None   # ("2", from, to): a "STAGE 2 CONSTRUCTION" dimension


def _interp(poly, x):
    xs, ys = zip(*poly)
    return np.interp(x, xs, ys, left=np.nan, right=np.nan)


def true_areas(spec: SectionSpec) -> tuple[float, float]:
    """Exact fill and cut (ft^2) between the two polylines."""
    x0 = max(spec.ground[0][0], spec.design[0][0])
    x1 = min(spec.ground[-1][0], spec.design[-1][0])
    x = np.linspace(x0, x1, 200001)
    d = _interp(spec.design, x) - _interp(spec.ground, x)     # +ve: design above ground = fill
    dx = x[1] - x[0]
    return float(np.clip(d, 0, None).sum() * dx), float(np.clip(-d, 0, None).sum() * dx)


def true_side_areas(spec: SectionSpec) -> dict:
    """Exact fill and cut (ft^2) left (offset < 0) and right of the centreline."""
    x0 = max(spec.ground[0][0], spec.design[0][0])
    x1 = min(spec.ground[-1][0], spec.design[-1][0])
    x = np.linspace(x0, x1, 200001)
    d = _interp(spec.design, x) - _interp(spec.ground, x)
    dx = x[1] - x[0]
    out = {}
    for side, keep in (("left", x < 0), ("right", x >= 0)):
        out[side] = {"fill_ft2": float(np.clip(d[keep], 0, None).sum() * dx),
                     "cut_ft2": float(np.clip(-d[keep], 0, None).sum() * dx)}
    return out


def embankment(station: str, ground_elev: float, height: float, top_half: float = 18.0,
               slope: float = 2.0, tilt: float = 0.0) -> SectionSpec:
    """A road on fill: flat-ish ground, template with 2:1 side slopes."""
    g = [(-140, ground_elev - 140 * tilt), (-40, ground_elev - 40 * tilt + 0.4), (25, ground_elev + 25 * tilt - 0.3),
         (140, ground_elev + 140 * tilt)]
    top = ground_elev + height
    run = height * slope
    d = [(-top_half - run, ground_elev - (top_half + run) * tilt), (-top_half, top), (0, top + 0.36),
         (top_half, top), (top_half + run, ground_elev + (top_half + run) * tilt)]
    return SectionSpec(station, g, d)


def cut_section(station: str, ground_elev: float, depth: float) -> SectionSpec:
    """A road in cut with ditches: ground above the template."""
    g = [(-140, ground_elev), (-60, ground_elev + 1.0), (60, ground_elev + 1.5), (140, ground_elev + 1.0)]
    base = ground_elev - depth
    d = [(-24 - depth * 2, ground_elev + 1.0 - 0.0), (-24, base - 1.5), (-20, base - 1.5), (-16, base),
         (16, base), (20, base - 1.5), (24, base - 1.5), (24 + depth * 2 + 3, ground_elev + 1.4)]
    return SectionSpec(station, g, d)


def walled_cut(station: str, ground_elev: float, depth: float, wall_at: float = -30.0) -> SectionSpec:
    """A deep cut retained by a wall on the left: the back slope comes down
    2:1 to a bench at the wall's top, the wall face drops to the roadway, and
    the right side climbs out 2:1 -- the template's deep end is the wall foot."""
    g = [(-140, ground_elev), (-40, ground_elev + 1.0), (40, ground_elev - 2.0), (140, ground_elev - 3.0)]
    base = ground_elev - depth
    bench = ground_elev - depth / 2
    top_l = wall_at - 2 * (ground_elev + 0.8 - bench)
    right = 24 + 2 * depth
    d = [(top_l, ground_elev + 0.8 + (top_l + 40) * 0), (wall_at, bench), (wall_at + 0.01, base),
         (24, base + 0.5), (right, ground_elev - 2.0 - (right - 40) * 0.01)]
    return SectionSpec(station, g, d)


def fill_wall(station: str, ground_elev: float, height: float, wall_at: float = 24.0) -> SectionSpec:
    """A road on fill held by a vertical wall on the right: the roadway runs
    flat to the wall's top, the face drops straight to the ground."""
    g = [(-140, ground_elev), (-40, ground_elev + 0.4), (40, ground_elev - 0.3), (140, ground_elev - 0.5)]
    top = ground_elev + height
    foot = float(_interp(g, wall_at))
    toe = -18 - 2 * height
    d = [(toe, float(_interp(g, toe))), (-18, top), (0, top + 0.36), (wall_at, top), (wall_at + 0.01, foot)]
    return SectionSpec(station, g, d)


def curbed_street(station: str, ground_elev: float, depth: float, curb: float) -> SectionSpec:
    """A street in cut between vertical curbs, sidewalks behind them sloping
    up to the ground."""
    g = [(-140, ground_elev), (-60, ground_elev + 0.5), (60, ground_elev + 0.8), (140, ground_elev + 0.6)]
    base = ground_elev - depth
    d = [(-30, float(_interp(g, -30))), (-24, base + curb + 0.12), (-18, base + curb), (-17.99, base),
         (0, base + 0.3), (17.99, base), (18, base + curb), (24, base + curb + 0.12), (30, float(_interp(g, 30)))]
    return SectionSpec(station, g, d)


def rolling_section(station: str, elev: float, amp: float = 2.0, phase: float = 0.0) -> SectionSpec:
    """Rolling ground that crosses a flat template several times, so one
    section has several separate cut zones and fill zones."""
    xs = np.arange(-140, 141, 10.0)
    g = [(float(x), float(elev + amp * np.sin((x + phase) / 9.0))) for x in xs]
    d = [(-62, elev - 1.5), (-56, elev), (0, elev + 1.12), (56, elev), (62, elev - 1.5)]   # 2 % crown
    return SectionSpec(station, g, d)


def zones(spec: SectionSpec) -> tuple[int, int]:
    """Number of separate fill zones and cut zones in a section."""
    x0 = max(spec.ground[0][0], spec.design[0][0])
    x1 = min(spec.ground[-1][0], spec.design[-1][0])
    x = np.linspace(x0, x1, 20001)
    sign = np.sign(_interp(spec.design, x) - _interp(spec.ground, x))
    runs = [v for i, v in enumerate(sign) if v != 0 and (i == 0 or sign[i - 1] != v)]
    return runs.count(1.0), runs.count(-1.0)


@dataclass
class Grid:
    """How a sheet is drawn: ft per major grid square on each axis and
    points per ft on each axis (vertical exaggeration when they differ)."""
    ft_x: float = GRID_FT
    ft_y: float = GRID_FT
    pt_x: float = PT_PER_FT
    pt_y: float = PT_PER_FT
    dash_by_dash: bool = False         # draw the ground's dashes as separate strokes (some CAD exports)


def _draw_section(page: fitz.Page, spec: SectionSpec, y_axis: float, x_c: float,
                  elev0: float, n_squares: int, grid: Grid = Grid()) -> None:
    sx = lambda off: x_c + off * grid.pt_x
    sy = lambda el: y_axis - (el - elev0) * grid.pt_y
    top = y_axis - n_squares * grid.ft_y * grid.pt_y
    left, right = sx(-140), sx(140)
    shape = page.new_shape()
    for off in np.arange(-140, 141, grid.ft_x):                    # vertical grid
        shape.draw_line((sx(off), top), (sx(off), y_axis))
    for k in range(n_squares + 1):                                 # horizontal grid
        y = y_axis - k * grid.ft_y * grid.pt_y
        shape.draw_line((left, y), (right, y))
    shape.finish(width=0.48, color=(0, 0, 0), closePath=False)
    tick = 1.0 if grid.ft_x <= 10 else 2.0
    for off in np.arange(-140, 140, tick):                        # axis ticks
        if off % grid.ft_x:
            shape.draw_line((sx(off), y_axis), (sx(off), y_axis - 1.6))
    shape.finish(width=0.24, color=(0, 0, 0), closePath=False)
    for a, b, t in spec.pavement:                                  # existing asphalt: a solid band on the ground
        xs = np.linspace(a, b, 12)
        top = [(sx(x), sy(float(_interp(spec.ground, x)))) for x in xs]
        bot = [(sx(x), sy(float(_interp(spec.ground, x)) - t)) for x in xs[::-1]]
        shape.draw_polyline(top + bot + top[:1])
    for x in spec.poles:                                           # a pole: upright line with a cross arm
        g = float(_interp(spec.ground, x))
        if spec.pole_style == "box":
            shape.draw_rect(fitz.Rect(sx(x - 0.5), sy(g + 25), sx(x + 0.5), sy(g - 6)))
            shape.draw_line((sx(x - 3), sy(g + 23)), (sx(x + 3), sy(g + 23)))
            shape.draw_line((sx(x), sy(g + 22)), (sx(x + 12), sy(float(_interp(spec.ground, x + 12)))))
            continue
        shape.draw_line((sx(x), sy(g)), (sx(x), sy(g + 14)))
        shape.draw_line((sx(x - 2), sy(g + 12.5)), (sx(x + 2), sy(g + 12.5)))
    if spec.pavement or spec.poles:
        shape.finish(width=0.6, color=(0, 0, 0), closePath=False)       # drawn solid, like the template
    if grid.dash_by_dash:                                          # existing ground drawn dash by dash
        shape.finish(width=0.6, color=(0, 0, 0), closePath=False)
        dashes = page.new_shape()
        xs = np.arange(spec.ground[0][0], spec.ground[-1][0], 1.1)    # 0.7 ft dashes, 0.4 ft gaps
        for a in xs:
            b = min(a + 0.7, spec.ground[-1][0])
            dashes.draw_line((sx(a), sy(float(_interp(spec.ground, a)))), (sx(b), sy(float(_interp(spec.ground, b)))))
            dashes.finish(width=0.84, color=(0, 0, 0), closePath=False)
        dashes.commit()
    else:
        shape.draw_polyline([(sx(o), sy(e)) for o, e in spec.ground])  # existing ground: dashed
        shape.finish(width=0.84, color=(0, 0, 0), dashes="[2.52 1.44] 0", closePath=False)
    shape.draw_polyline([(sx(o), sy(e)) for o, e in spec.design])  # proposed design: solid
    shape.finish(width=0.6, color=(0, 0, 0), closePath=False)
    shape.commit()
    for off in np.arange(-140, 141, grid.ft_x):
        page.insert_text((sx(off) - 6, y_axis + 9), str(int(off)), fontsize=6)
    for k in range(n_squares + 1):
        page.insert_text((right + 4, y_axis - k * grid.ft_y * grid.pt_y + 2), f"{elev0 + k * grid.ft_y:g}", fontsize=6)
    page.insert_text((sx(118), y_axis - 0.8 * grid.ft_y * grid.pt_y), spec.station, fontsize=11)
    if spec.stage_mark:
        sid, a, b = spec.stage_mark
        y = top + 0.6 * grid.ft_y * grid.pt_y
        dim = page.new_shape()
        dim.draw_line((sx(a), y), (sx(b), y))
        dim.draw_line((sx(a), y - 4), (sx(a), y + 4))
        dim.draw_line((sx(b), y - 4), (sx(b), y + 4))
        dim.finish(width=0.4, color=(0, 0, 0), closePath=False)
        dim.commit()
        page.insert_text((sx((a + b) / 2) - 65, y + 13), f"STAGE {sid} CONSTRUCTION", fontsize=10)


def make_pdf(path: str, pages: list[list[SectionSpec]], scanned: bool = False, dpi: int = 254,
             grid: Grid = Grid(), squares: int | None = None, title: str | None = None,
             border_below: bool = False) -> None:
    """Stack the sections of each inner list on one 17x11 in sheet, highest
    station at the top (the usual convention)."""
    doc = fitz.open()
    for specs in pages:
        page = doc.new_page(width=1224, height=792)
        n_squares = max(4, int(round(4 * GRID_FT / grid.ft_y)) if grid.ft_y < GRID_FT else 4)
        n_squares = squares or min(n_squares, int(140 // (grid.ft_y * grid.pt_y)))
        y = 120.0
        for spec in specs:
            y_axis = y + n_squares * grid.ft_y * grid.pt_y
            lo = min(e for _, e in spec.ground + spec.design)
            elev0 = grid.ft_y * np.floor(lo / grid.ft_y) - grid.ft_y
            _draw_section(page, spec, y_axis, 612, elev0, n_squares, grid)
            last_axis = y_axis
            y = y_axis + 40
        if border_below:                                           # a title block's border one square below
            yb = last_axis + grid.ft_y * grid.pt_y
            page.draw_line((40, yb), (1184, yb), width=1.5)
        if title:                                                  # title block, bottom right
            page.insert_text((1000, 760), title, fontsize=14)
    if scanned:
        out = fitz.open()
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            p2 = out.new_page(width=page.rect.width, height=page.rect.height)
            p2.insert_image(p2.rect, pixmap=pix)
        out.save(path)
    else:
        doc.save(path)
