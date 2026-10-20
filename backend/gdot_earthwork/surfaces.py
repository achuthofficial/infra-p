"""
Extract the two surfaces of one cross-section -- EXISTING GROUND (dashed)
and PROPOSED DESIGN (solid) -- and measure the fill/cut area between them.

Two extractors produce the same `Surfaces` result:

* raster_surfaces(): for scanned sheets. Works from pixels.
* vector_surfaces(): for CAD-exported PDFs. Reads the drawing's own stroke
  geometry, so dashed vs solid, text vs line and grid vs data are known
  exactly instead of guessed from pixels.

Both resample the frame to a CANONICAL resolution (CANON_PPF pixels per
foot, the same on both axes) before tracing. Scans arrive at 14.4 px/ft
(19 series), 18 px/ft (23 series) or anything else; normalising once
means every pixel constant in tracing.py means the same thing on every
sheet, and vertical exaggeration (different horizontal and vertical
scales) is undone for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PipelineConfig
from .frames import Frame
from .pages import PageImage
from .tracing import (
    best_path,
    candidates,
    climb_from_wall,
    denoise,
    densify,
    despike,
    drop_rulers,
    drop_structures,
    fill_along_gridlines,
    ground_candidates,
    interp_spans,
    refine_gaps,
    snap_bottom,
    snap_top,
    split_chain,
    split_dash_solid,
    strip_grid,
    template_strokes,
    trace_ground,
    trim_buried_tail,
)

CANON_PPF = 14.4   # px per ft the tracing constants were tuned at (19-series native)

# How far (canonical px) the surface tracer may jump across a gap. Existing
# ground routinely disappears for 50-100 ft where it runs underneath a
# thick proposed-design line (mill & overlay, widening); the tracer must
# bridge that rather than keep only the longer half of the ground.
TRACE_WINDOW = int(110 * CANON_PPF)

# Template strokes closer than this belong to one network; isolated solid
# pieces narrower than MIN_TEMPLATE_PX are annotation (leaders, callouts).
LINK_PX = int(2.0 * CANON_PPF)

# Reward for the ground tracer following a template stroke instead of its own
# dashes (see tracing.ground_candidates).
HIDDEN_GROUND_WEIGHT = 1 / 3
MIN_TEMPLATE_PX = int(15 * CANON_PPF)

# A traced point this close (canonical px) to a grid row may be a line that
# ran on along it and was erased with the grid.
GRID_TOL_PX = 9.0


@dataclass
class Surfaces:
    ground: np.ndarray          # row (canonical px, frame-local) per column, NaN = absent
    design: np.ndarray          # finished grade (top of the template)
    width: int                  # canonical columns
    height: int                 # canonical rows
    source: str                 # "raster" or "vector"
    roi: np.ndarray | None = None          # cleaned ink mask, for overlays
    max_gap_ground_ft: float = 0.0
    max_gap_design_ft: float = 0.0
    trimmed: list[dict] = field(default_factory=list)
    structures: list[dict] = field(default_factory=list)   # design spans dropped as bridges
    subgrade: np.ndarray | None = None     # bottom of the pavement structure (= design elsewhere)
    other_pen_ft: float = 0.0              # template drawn in another stage's (grey) pen, left out

    def surface(self, which: str) -> np.ndarray:
        if which == "subgrade" and self.subgrade is not None:
            return self.subgrade
        return self.design

    @property
    def overlap(self) -> np.ndarray:
        return ~np.isnan(self.ground) & ~np.isnan(self.design)


@dataclass
class Area:
    fill_ft2: float
    cut_ft2: float


def measure(s: Surfaces, surface: str = "subgrade") -> Area:
    """Fill where the proposed surface sits above existing ground (smaller row
    = higher), cut where it sits below. `surface`: "subgrade" (bottom of the
    pavement structure -- the usual earthwork convention) or "finished"
    (finished grade). Columns are 1/CANON_PPF ft wide and the curves are
    piecewise linear per column, so a column sum is exact to the raster."""
    top = s.surface(surface)
    ov = ~np.isnan(s.ground) & ~np.isnan(top)
    if not ov.any():
        return Area(0.0, 0.0)
    d = (s.ground[ov] - top[ov]) / CANON_PPF        # ft, +ve = fill
    col_ft = 1.0 / CANON_PPF
    return Area(float(d[d > 0].sum() * col_ft), float(-d[d < 0].sum() * col_ft))


def area_strips(s: Surfaces, surface: str = "subgrade") -> tuple[np.ndarray, np.ndarray]:
    """Fill and cut area (ft²) in each 1 ft wide strip across the section,
    counted from the frame's left edge -- the same column integration as
    `measure`, so the strips add up to its totals. Splitting them at the
    centreline gives the left and right areas."""
    top = s.surface(surface)
    n = int(np.ceil(s.width / CANON_PPF))
    ov = ~np.isnan(s.ground) & ~np.isnan(top)
    if not ov.any():
        return np.zeros(n), np.zeros(n)
    cols = np.where(ov)[0]
    d = (s.ground[cols] - top[cols]) / CANON_PPF / CANON_PPF     # ft² in each column, +ve = fill
    strip = (cols // CANON_PPF).astype(int)
    return (np.bincount(strip, np.where(d > 0, d, 0), n), np.bincount(strip, np.where(d < 0, -d, 0), n))


# Design lines are solid: a stretch with no ink this long is a real break
# (bridge end, drawing break), not a dash gap, and is never bridged.
DESIGN_SPLIT_FT = 5.0


# How far below finished grade the subgrade (bottom of the pavement box) may
# lie. Pavement sections and curbs are 1-3 ft deep.
SUBGRADE_REACH_FT = 3.5


def _finish(ground_px: dict, design_px: dict, solid: np.ndarray, roi: np.ndarray,
            W: int, H: int, cfg: PipelineConfig, source: str, refine: bool) -> Surfaces:
    if refine:
        ground_px, _ = refine_gaps(ground_px, roi, CANON_PPF, cfg.gap_refine_ft, cfg.gap_tol_px)
        design_px, _ = refine_gaps(design_px, solid, CANON_PPF, cfg.gap_refine_ft, cfg.gap_tol_px)
    _, gap_g = interp_spans(ground_px, W, CANON_PPF, cfg.gap_warn_ft)
    G, _ = despike(densify(ground_px, W), cfg.despike_win, cfg.despike_px)

    pieces = split_chain(design_px, int(DESIGN_SPLIT_FT * CANON_PPF))
    pieces, structures = drop_structures(pieces, G, CANON_PPF)
    D = np.full(W, np.nan)
    B = np.full(W, np.nan)
    gap_d = 0.0
    reach = int(SUBGRADE_REACH_FT * CANON_PPF)
    for pc in pieces:
        for arr, chain in ((D, pc), (B, snap_bottom(solid, pc, reach))):
            dense = densify(chain, W)
            ok = ~np.isnan(dense)
            arr[ok] = dense[ok]
        gap_d = max(gap_d, interp_spans(pc, W, CANON_PPF, cfg.gap_warn_ft)[1])
    D, _ = despike(D, cfg.despike_win, cfg.despike_px)
    B, _ = despike(B, cfg.despike_win, cfg.despike_px)
    # a wall: carry the template on up its back slope to the ground (no pavement there)
    walled, found = climb_from_wall(D, G, solid > 0, CANON_PPF)
    if not found and roi is not None:          # a thin back slope may not have passed as a solid stroke
        walled, found = climb_from_wall(D, G, roi > 0, CANON_PPF)
    added = np.isnan(D) & ~np.isnan(walled)
    D[added] = walled[added]
    B[added] = walled[added]
    x = (np.arange(W) - W / 2) / CANON_PPF
    D, trimmed = trim_buried_tail(x, G, D, CANON_PPF, cfg.buried_run_ft, cfg.buried_depth_ft, subgrade=B)
    B[np.isnan(D)] = np.nan
    s = Surfaces(G, D, W, H, source, roi, gap_g, gap_d, trimmed)
    s.subgrade = B
    s.structures = [dict(x_from=round(float(x[n["x_from"]]), 1), x_to=round(float(x[n["x_to"]]), 1))
                    for n in structures]
    return s


def _canon_size(frame: Frame, cfg: PipelineConfig) -> tuple[float, float, int, int]:
    # frame.ft_x / ft_y: ft per grid square as calibrated (stages.calibrate)
    sx = CANON_PPF / (frame.pitch_x / frame.ft_x)
    sy = CANON_PPF / (frame.pitch_y / frame.ft_y)
    W = int(round((frame.right - frame.left + 1) * sx))
    H = int(round((frame.y1 - frame.y0 + 1) * sy))
    return sx, sy, W, H


# ------------------------------------------------------------------ raster

def _blank_columns(mask: np.ndarray, frame: Frame, sx: float, exclude, erase=None, sy: float = 1.0) -> np.ndarray:
    """`mask` with the columns of the page-pixel x ranges in `exclude`
    cleared, and the annotation lines (x_from, x_to, y) in `erase` removed."""
    if not exclude and not erase:
        return mask
    out = mask.copy()
    for a, b in exclude or []:
        c0, c1 = max(0, int((a - frame.left) * sx)), min(mask.shape[1], int((b - frame.left) * sx) + 1)
        out[:, c0:c1] = 0
    for a, b, y in erase or []:
        c0, c1 = max(0, int((a - frame.left) * sx) - 3), min(mask.shape[1], int((b - frame.left) * sx) + 4)
        r = int((y - frame.y0) * sy)
        out[max(0, r - 4):r + 5, c0:c1] = 0
    return out


def own_pen(solid: np.ndarray, gray: np.ndarray, W: int, H: int, black: int = 25,
            grey: tuple[int, int] = (60, 200)) -> tuple[np.ndarray, float]:
    """A staged sheet draws its own stage's work in black and the other
    stages' (and the traffic phases') in a grey pen. Where a section's
    template has both, keep the black strokes only. `solid`: canonical
    template mask; `gray`: the frame's page pixels. Returns (mask, ft of
    grey template left out where no black one is drawn) -- the mask
    unchanged when the section is drawn in one pen."""
    k = max(3, int(np.ceil(max(gray.shape[1] / W, gray.shape[0] / H))) + 2)
    dark = cv2.resize(cv2.erode(gray, np.ones((k, k), np.uint8)), (W, H), interpolation=cv2.INTER_NEAREST)
    on = solid > 0
    # a stroke's shade: the mean over the ink around each pixel, so the odd
    # dark speck in a grey line (or a light edge of a black one) doesn't decide
    n = cv2.blur(on.astype(np.float32), (9, 9))
    shade = np.where(on, cv2.blur(np.where(on, dark, 0).astype(np.float32), (9, 9)) / np.maximum(n, 1e-6), 255)
    is_black = on & (shade < black)
    is_grey = on & (shade >= black) & (shade < grey[1])
    black_cols, grey_cols = is_black.any(axis=0), is_grey.any(axis=0)
    if black_cols.sum() < MIN_TEMPLATE_PX or grey_cols.sum() < MIN_TEMPLATE_PX:
        return solid, 0.0
    out = solid.copy()
    out[is_grey] = 0
    return out, float((grey_cols & ~black_cols).sum()) / CANON_PPF


def raster_surfaces(frame: Frame, img: PageImage, cfg: PipelineConfig, exclude=None, erase=None,
                    stage_pen: bool = False) -> Surfaces:
    """`exclude`: page-pixel x ranges whose proposed lines are not this
    sheet's to measure (another construction stage): the design is traced
    without them. `stage_pen`: a staged plan set -- template drawn in the
    grey pen of other stages is left out (own_pen)."""
    x0, x1, y0, y1 = frame.left, frame.right + 1, frame.y0, frame.y1 + 1
    gray = img.gray[y0:y1, x0:x1]
    ink = (gray < cfg.ink_thresh).astype(np.uint8)
    # Grid lines are removed at native resolution, where their positions are
    # known to the pixel, then the cleaned mask is resampled.
    V = [(a - x0, b - x0) for a, b in frame.vlines]
    Hh = [(a - y0, b - y0) for a, b in frame.hlines]
    V += _partial_grid_lines(ink, V, frame.pitch_x, axis=1)
    Hh += _partial_grid_lines(ink, Hh, frame.pitch_y, axis=0)
    clean = strip_grid(ink, V, Hh, sliver=int(0.8 * min(frame.pitch_x, frame.pitch_y)))
    sx, sy, W, H = _canon_size(frame, cfg)
    canon = cv2.resize(clean.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    roi = (canon > 0.25).astype(np.uint8)
    # blank the outermost band: frame border, axis ticks
    m = 3
    roi[:m], roi[-m:], roi[:, :m], roi[:, -m:] = 0, 0, 0, 0
    roi, _ = denoise(roi)
    dash, solid, _, _ = split_dash_solid(roi)
    solid = drop_rulers(solid, int(0.4 * W))
    solid = template_strokes(solid, LINK_PX, MIN_TEMPLATE_PX)
    solid = _drop_detached_in_margins(solid, (frame.top - y0) * sy, (frame.bottom - y0) * sy)
    ground_px = trace_ground(dash, solid, HIDDEN_GROUND_WEIGHT, TRACE_WINDOW, CANON_PPF)
    grid_rows = [((a + b) / 2 - y0) * sy for a, b in frame.hlines]

    def trace(mask):
        d = trace_design(mask, ground_px, grid_rows=grid_rows)
        return fill_along_gridlines(d, grid_rows, int(1.0 * CANON_PPF), GRID_TOL_PX)

    design_mask, other_ft = own_pen(solid, gray, W, H) if stage_pen else (solid, 0.0)
    surf = _finish_with_retry(ground_px, trace, _blank_columns(design_mask, frame, sx, exclude, erase, sy), roi, W, H,
                              cfg, "raster", refine=True)
    surf.other_pen_ft = other_ft
    return surf


def _drop_detached_in_margins(mask: np.ndarray, top_row: float, bottom_row: float,
                              rect_frac: float = 0.7) -> np.ndarray:
    """The frame is padded past its top and bottom grid lines so a ditch drawn
    into the label strip is still measured; on some sheets the padding reaches
    a title block, whose scale bar and table rules are long solid strokes. A
    stroke lying entirely in the padding and drawn mostly of horizontal and
    vertical rules (at least `rect_frac` of its ink) is a box, not part of the
    section: a ditch below the axis is drawn with sloped sides."""
    m = mask.astype(np.uint8)
    run = int(1.5 * CANON_PPF)
    rect = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1))) | \
        cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = mask.copy()
    for k in range(1, n):
        y, h = st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_HEIGHT]
        # wholly in a margin, or hanging off the grid's edge line into it
        below = y >= bottom_row - 0.5 * CANON_PPF and y + h > bottom_row + 2
        above = y + h <= top_row + 0.5 * CANON_PPF and y < top_row - 2
        if not (below or above):
            continue
        comp = lab == k
        if rect[comp].sum() >= rect_frac * comp.sum():
            out[comp] = 0
    return out


def _erase_chain(mask: np.ndarray, chain: dict, half: int = 4) -> np.ndarray:
    """`mask` without the strokes a traced chain ran along."""
    band = np.zeros_like(mask)
    for x, y in chain.items():
        band[max(0, int(y) - half):int(y) + half + 1, max(0, x - 1):x + 2] = 1
    band = cv2.dilate(band, np.ones((3, 3), np.uint8))
    return mask & (1 - band)


def trace_design(mask: np.ndarray, ground: dict | None = None, rounds: int = 3,
                 grid_rows: list[float] | None = None) -> dict[int, float]:
    """The design line: the best chain through the template strokes, then
    any further template piece standing apart from it along the section (a
    widening on the far side of untouched existing pavement), each traced
    the same way. A chain only bridges gaps of a tracing window, so a second
    piece 60 ft away would otherwise be left out whole. Such a piece is graded
    ground: each of its ends ties into the existing ground (catch point, edge
    of pavement kept), which a note's underline and leader do not -- or
    carries the chain on along a grid line (`grid_rows`, scans): a flat
    roadbed at a round elevation is erased with the grid, which can split
    the template into pieces further apart than the tracing window."""
    chain = snap_top(mask, best_path(candidates(mask)))
    rest, last = mask.copy(), chain
    for _ in range(rounds):
        if not last:
            break
        rest[:, max(0, min(last) - LINK_PX):max(last) + LINK_PX + 1] = 0
        cols = np.flatnonzero(rest.any(axis=0))
        if cols.size == 0 or cols[-1] - cols[0] < MIN_TEMPLATE_PX:
            break
        last = snap_top(rest, best_path(candidates(rest)))
        if not last or max(last) - min(last) < MIN_TEMPLATE_PX:
            break
        if len(last) < 0.6 * (max(last) - min(last) + 1):
            break                       # hops between scattered marks (lettering), not a drawn line
        if ground is not None and not _ties_into(last, ground, chain, grid_rows):
            continue
        chain = {**chain, **{x: y for x, y in last.items() if x not in chain}}
    return dict(sorted(chain.items()))


def _ties_into(piece: dict, ground: dict, chain: dict | None = None, grid_rows: list[float] | None = None,
               tol_ft: float = 2.0) -> bool:
    """Both ends of a design piece lie on (within tol_ft of) the ground, or
    continue `chain` along a grid line (_along_grid)."""
    if not ground:
        return False
    gx = np.array(sorted(ground), float)
    gy = np.array([ground[x] for x in sorted(ground)], float)
    for x in (min(piece), max(piece)):
        if chain and grid_rows and _along_grid(piece, x, chain, grid_rows):
            continue
        if not gx[0] - CANON_PPF <= x <= gx[-1] + CANON_PPF:
            return False
        if abs(piece[x] - np.interp(x, gx, gy)) > tol_ft * CANON_PPF:
            return False
    return True


def _along_grid(piece: dict, end: int, chain: dict, grid_rows: list[float], tol: float = GRID_TOL_PX) -> bool:
    """The piece's `end` and the chain's nearest point beyond it both lie on
    the same grid row: the line ran on between them along the grid and was
    erased with it (fill_along_gridlines bridges such a gap)."""
    beyond = [x for x in chain if (x < end if end == min(piece) else x > end)]
    if not beyond:
        return False
    other = max(beyond) if end == min(piece) else min(beyond)
    rows = np.asarray(grid_rows, float)
    r = rows[np.argmin(np.abs(rows - piece[end]))]
    return abs(piece[end] - r) <= tol and abs(chain[other] - r) <= tol


def _finish_with_retry(ground_px, trace, solid, roi, W, H, cfg, source, refine) -> Surfaces:
    """Trace the design and finish. When the strongest solid line turns out
    to be a buried pipe or utility (rejected whole, leaving no design over
    the ground), erase it and trace again: the template is usually the next
    line down the list. The rejected run is still reported."""
    design_px = trace(solid)
    s = _finish(ground_px, design_px, solid, roi, W, H, cfg, source, refine)
    if s.overlap.any() or not s.trimmed or not design_px:
        return s
    rest = _erase_chain(solid, design_px)
    if not rest.any():
        return s
    again = _finish(ground_px, trace(rest), rest, roi, W, H, cfg, source, refine)
    if not again.overlap.any():
        return s
    again.trimmed = s.trimmed + again.trimmed
    return again


def _partial_grid_lines(ink: np.ndarray, bands: list[tuple[int, int]], pitch: float,
                        axis: int) -> list[tuple[int, int]]:
    """Grid lines drawn across only part of the plot (some sheets do this
    near the axes) are missed by frame detection, which looks for full-length
    lines, yet they would survive as long, perfectly straight "solid" strokes.
    The lattice is regular, so look on every expected grid position that has
    no detected line for a straight run at least two squares long.
    axis=0: horizontal lines (rows), axis=1: vertical lines (columns)."""
    if len(bands) < 2:
        return []
    centres = sorted((a + b) / 2 for a, b in bands)
    n = ink.shape[axis]
    expected, k0 = [], centres[0]
    k = int(np.floor(-k0 / pitch))
    while k0 + k * pitch < n:
        c = k0 + k * pitch
        if 0 <= c < n and min(abs(c - x) for x in centres) > 0.25 * pitch:
            expected.append(c)
        k += 1
    run = int(2 * pitch)
    shape = (run, 1) if axis == 0 else (1, run)       # cv2 kernel: (width, height)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape))
    prof = opened.sum(1 - axis)                         # ink along each row (axis 0) / column
    out = []
    for c in expected:
        lo, hi = max(0, int(c) - 4), min(n, int(c) + 5)
        hit = np.where(prof[lo:hi] >= run)[0]
        if len(hit):
            out.append((lo + int(hit.min()), lo + int(hit.max())))
    return out


# ------------------------------------------------------------------ vector

def _path_segments(d: dict) -> list[tuple]:
    segs = []
    for it in d["items"]:
        op = it[0]
        if op == "l":
            segs.append((it[1], it[2]))
        elif op == "c":
            pts = [it[1], it[2], it[3], it[4]]
            # flatten the Bezier into 8 chords
            prev = pts[0]
            for t in np.linspace(0, 1, 9)[1:]:
                a = (1 - t) ** 3
                b = 3 * (1 - t) ** 2 * t
                c = 3 * (1 - t) * t ** 2
                e = t ** 3
                p = (a * pts[0].x + b * pts[1].x + c * pts[2].x + e * pts[3].x,
                     a * pts[0].y + b * pts[1].y + c * pts[2].y + e * pts[3].y)
                segs.append((prev, p))
                prev = type(pts[0])(*p)
        elif op == "re":
            r = it[1]
            q = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
            segs += [(q[i], q[(i + 1) % 4]) for i in range(4)]
        elif op == "qu":
            q = it[1]
            pts = [q.ul, q.ur, q.lr, q.ll]
            segs += [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return [(_xy(a), _xy(b)) for a, b in segs]


def _xy(p) -> tuple[float, float]:
    return (p.x, p.y) if hasattr(p, "x") else (float(p[0]), float(p[1]))


def _is_dashed(d: dict) -> bool:
    dash = (d.get("dashes") or "").strip()
    return bool(dash) and not dash.startswith("[]")


def vector_strokes(frame: Frame, img: PageImage, cfg: PipelineConfig):
    """Dashed and solid data strokes inside the frame, in canonical frame-local
    pixels. Grid lines, tick marks and text glyphs are filtered out; filled
    pavement bands count as solid strokes (their outline)."""
    k = img.scale
    sx, sy, W, H = _canon_size(frame, cfg)
    fw, fh = frame.right - frame.left, frame.bottom - frame.top
    edge = 0.08 * frame.pitch_y
    dashed, solid, short, glyphs = [], [], [], []
    # text labels in page-image pixels: (x0, y0, x1, y1, is a word note)
    boxes = [(sp.x - hw, sp.y - hh, sp.x + hw, sp.y + hh, sum(c.isalpha() for c in sp.text) >= 3)
             for sp in img.spans if sp.text.strip() for hw, hh in [_half_box(sp)]]
    for d in img.page.get_drawings():
        kind = d.get("type") or ""
        if "s" not in kind and not _is_pavement_fill(d, k, frame):
            continue
        r = d["rect"]
        # page points -> (possibly rotated) page-image pixels
        (rx0, ry0), (rx1, ry1) = img.to_view(r.x0 * k, r.y0 * k), img.to_view(r.x1 * k, r.y1 * k)
        rx0, rx1 = sorted((rx0, rx1))
        ry0, ry1 = sorted((ry0, ry1))
        if any(bx0 <= rx0 and rx1 <= bx1 and by0 <= ry0 and ry1 <= by1 for bx0, by0, bx1, by1, _ in boxes):
            continue                        # lettering drawn as strokes over its own text (a bold label)
        if rx1 < frame.left or rx0 > frame.right or ry1 < frame.y0 or ry0 > frame.y1:
            continue
        if max(rx1 - rx0, ry1 - ry0) < 0.25 * frame.pitch_x:   # text glyphs, ticks, specks
            glyphs.append(((rx0 - frame.left) * sx, (ry0 - frame.y0) * sy, (rx1 - frame.left) * sx, (ry1 - frame.y0) * sy))
            segs = _path_segments(d)
            if len(segs) == 1 and "s" in kind:          # kept aside: may be one dash of a line drawn dash by dash
                (ax, ay), (bx, by) = segs[0]
                ax, ay = img.to_view(ax * k, ay * k)
                bx, by = img.to_view(bx * k, by * k)
                short.append(((ax - frame.left) * sx, (ay - frame.y0) * sy, (bx - frame.left) * sx, (by - frame.y0) * sy,
                              round(d.get("width") or 0, 2), tuple(round(c, 2) for c in (d.get("color") or ()))))
            continue
        is_dash = _is_dashed(d)
        if not is_dash and _is_landed_leader(d, img, k, boxes, frame.pitch_x / frame.ft_x):
            continue
        for (ax, ay), (bx, by) in _path_segments(d):
            ax, ay = img.to_view(ax * k, ay * k)
            bx, by = img.to_view(bx * k, by * k)
            # near-flat, not exactly flat: CAD exports often store grid lines
            # with a pixel of drift over their full length
            horiz = abs(ay - by) < max(0.2, 0.002 * abs(ax - bx))
            vert = abs(ax - bx) < max(0.2, 0.002 * abs(ay - by))
            # grid & frame lines span most of the frame; the frame's own top
            # and bottom (offset axis) lines are often drawn in short pieces
            # (grid lines are always solid -- a long flat stretch of dashed
            # existing ground must never be mistaken for one)
            if horiz and not is_dash and (abs(ax - bx) > 0.5 * fw
                                          or min(abs(ay - frame.top), abs(ay - frame.bottom)) < 3):
                continue
            if vert and abs(ay - by) > 0.5 * fh and not is_dash:
                continue
            # axis tick marks: short solid strokes hugging the frame edge
            # (often drawn as one path spanning the whole axis)
            if not is_dash and max(_edge_dist(frame, ax, ay), _edge_dist(frame, bx, by)) < edge:
                continue
            if not is_dash and _is_leader(ax, ay, bx, by, boxes, frame.pitch_x / frame.ft_x):
                continue
            seg = ((ax - frame.left) * sx, (ay - frame.y0) * sy,
                   (bx - frame.left) * sx, (by - frame.y0) * sy)
            (dashed if is_dash else solid).append(seg)
    if not dashed and short:
        # no dashed line style at all: some CAD exports draw a dashed line
        # dash by dash, as separate little segments
        dashed = _dash_chains(short)
    for sp in img.spans:                                  # lettering from the text layer
        half = 0.3 * len(sp.text.strip()) * sp.size
        glyphs.append(((sp.x - half - frame.left) * sx, (sp.y - sp.size / 2 - frame.y0) * sy,
                       (sp.x + half - frame.left) * sx, (sp.y + sp.size / 2 - frame.y0) * sy))
    solid = [s_ for s_ in solid if not _is_underline(s_, glyphs)]
    return dashed, solid, W, H


def _half_box(sp) -> tuple[float, float]:
    """Half width and height of a word on the image: its box from the text
    layer, else estimated from its length (read as level text)."""
    if sp.w > 0 and sp.h > 0:
        return sp.w / 2 + 1, sp.h / 2 + 1
    return 0.3 * len(sp.text.strip()) * sp.size, 0.6 * sp.size


def _is_landed_leader(d: dict, img: PageImage, k: float, boxes: list, ppf: float) -> bool:
    """A leader drawn the usual CAD way: one path from the point it marks to
    a short level landing that stops just short of a word note. The slanted
    part can rise from the template's catch point, so it must go as a whole."""
    segs = _path_segments(d)
    if not 2 <= len(segs) <= 3:
        return False
    for (ax, ay), (bx, by) in (segs[-1], (segs[0][1], segs[0][0])):
        (ax, ay), (bx, by) = img.to_view(ax * k, ay * k), img.to_view(bx * k, by * k)
        if abs(by - ay) > 0.2 * abs(bx - ax) or abs(bx - ax) > 4 * ppf:
            continue                                  # not a short level landing
        for bx0, by0, bx1, by1, word in boxes:
            gap = max(1.0 * ppf, 0.6 * (by1 - by0))
            if word and by0 <= by <= by1 and (bx0 - gap <= bx <= bx0 + 1 or bx1 - 1 <= bx <= bx1 + gap):
                return True
    return False


def _is_leader(ax: float, ay: float, bx: float, by: float, boxes: list, ppf: float) -> bool:
    """A leader line: a steep stroke ending at a word note ("EDGE OF STREAM
    BANK", "LIMIT OF CONSTRUCTION"), drawn from the point it describes. It
    may touch the template, so it would otherwise be traced as part of it."""
    if abs(by - ay) < abs(bx - ax):                  # level strokes are underlines' business
        return False
    pad = 0.6 * ppf
    for x, y in ((ax, ay), (bx, by)):
        for bx0, by0, bx1, by1, word in boxes:
            if word and bx0 - pad <= x <= bx1 + pad and by0 - pad <= y <= by1 + pad:
                return True
    return False


def _is_underline(seg: tuple, glyphs: list[tuple], cover: float = 0.6) -> bool:
    """A level line with lettering just above it along most of its length is
    a label's underline or leader (a utility note: "EX. 8" SANITARY SEWER"),
    not part of the template. A pavement top with a "2.0%" label has only a
    few letters over a long line, so it stays."""
    x0, y0, x1, y1 = seg
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    length = x1 - x0
    if length < 3 * CANON_PPF or abs(y1 - y0) > 0.1 * CANON_PPF:
        return False
    y = (y0 + y1) / 2
    covered = np.zeros(int(length) + 1, bool)
    for gx0, gy0, gx1, gy1 in glyphs:
        if 0 <= y - gy1 <= 2.0 * CANON_PPF and gy1 - gy0 <= 2.5 * CANON_PPF:      # just above the line
            a, b = max(0, int(gx0 - x0)), min(len(covered), int(gx1 - x0) + 1)
            covered[a:b] = True
    return covered.mean() >= cover


def _dash_chains(short: list[tuple], min_dashes: int = 5, min_span_ft: float = 10.0) -> list[tuple]:
    """Segments that are dashes of one line: short single strokes of the same
    pen, each starting a small gap after the previous one ends and heading the
    same way, in a run of at least `min_dashes` spanning `min_span_ft`.
    Letters (drawn as strokes when a PDF has no text) and tick marks do not
    line up like that. Canonical pixels in, (x0, y0, x1, y1) out."""
    ppf = CANON_PPF
    cand = [c for c in short if 0.3 * ppf <= np.hypot(c[2] - c[0], c[3] - c[1]) <= 4.0 * ppf]
    n = len(cand)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    unit = []
    for x0, y0, x1, y1, *_ in cand:
        L = np.hypot(x1 - x0, y1 - y0)
        unit.append(((x1 - x0) / L, (y1 - y0) / L))
    for i in range(n):
        a = cand[i]
        for j in range(i + 1, n):
            b = cand[j]
            if a[4:] != b[4:]:                                  # same pen: width and colour
                continue
            if abs(unit[i][0] * unit[j][0] + unit[i][1] * unit[j][1]) < 0.8:     # same heading (within ~35 deg)
                continue
            gap = min(np.hypot(p[0] - q[0], p[1] - q[1])
                      for p in ((a[0], a[1]), (a[2], a[3])) for q in ((b[0], b[1]), (b[2], b[3])))
            if 0.05 * ppf <= gap <= 3.5 * ppf:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        xs = [v for i in members for v in (cand[i][0], cand[i][2])]
        if len(members) >= min_dashes and max(xs) - min(xs) >= min_span_ft * ppf:
            out += [cand[i][:4] for i in members]
    return out


def _is_pavement_fill(d: dict, k: float, frame: Frame) -> bool:
    """Some CAD exports draw the pavement as a filled dark band with no
    outline stroke. Its outline is the finished grade on top and the
    subgrade below, so it is read as solid linework: a dark fill that is
    thin (under 3 ft) and at least half a grid square long."""
    fill = d.get("fill")
    if "f" not in (d.get("type") or "") or not fill or max(fill) > 0.35:
        return False
    r = d["rect"]
    return r.height * k < 0.3 * frame.pitch_y and r.width * k >= 0.5 * frame.pitch_x


def _edge_dist(frame: Frame, x: float, y: float) -> float:
    return min(abs(y - frame.top), abs(y - frame.bottom), abs(x - frame.left), abs(x - frame.right))


def _draw(segs, W, H) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in segs:
        cv2.line(m, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, 2)
    m[:2], m[-2:], m[:, :2], m[:, -2:] = 0, 0, 0, 0
    return m


def vector_surfaces(frame: Frame, img: PageImage, cfg: PipelineConfig, exclude=None, erase=None) -> Surfaces | None:
    if img.page is None:
        return None
    dashed, solid, W, H = vector_strokes(frame, img, cfg)
    if len(dashed) < 3 or len(solid) < 2:
        return None          # this sheet doesn't draw ground dashed -- fall back to raster
    dmask, smask = _draw(dashed, W, H), _draw(solid, W, H)
    smask = template_strokes(smask, LINK_PX, MIN_TEMPLATE_PX)
    sy = _canon_size(frame, cfg)[1]
    smask = _drop_detached_in_margins(smask, (frame.top - frame.y0) * sy, (frame.bottom - frame.y0) * sy)
    gc, gw = ground_candidates(dmask, smask, HIDDEN_GROUND_WEIGHT)
    ground_px = best_path(gc, win=TRACE_WINDOW, weights=gw)
    return _finish_with_retry(ground_px, lambda m: trace_design(m, ground_px),
                              _blank_columns(smask, frame, _canon_size(frame, cfg)[0], exclude, erase,
                                             _canon_size(frame, cfg)[1]), dmask | smask, W, H, cfg, "vector", refine=False)
