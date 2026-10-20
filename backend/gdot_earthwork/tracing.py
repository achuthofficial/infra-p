"""
Low-level raster tracing primitives: grid stripping, glyph removal,
dash/solid separation, and tracing a surface through noisy, dashed,
gapped ink.

These operate on a binary ink mask at the pipeline's CANONICAL resolution
(see surfaces.CANON_PPF) -- every pixel constant below was tuned there,
so callers resample first instead of rescaling each threshold.

Applied to both surfaces:
    candidates()        per-column ink midpoints
    best_path()         max-scoring DP chain through those candidates
    snap_top()          (design only) recentre on the topmost stroke
    refine_gaps()       recall pass: look for missed dashes inside big gaps
    despike()/densify() outlier rejection + dense per-column array
    trim_buried_tail()  drop false "cut" caused by buried services
    climb_from_wall()   (design only) the back slope above a wall, back to ground
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter


def runs1d(a: np.ndarray, tol: int = 3) -> list[tuple[int, int]]:
    """Group a sorted index array into (start, end) runs, merging gaps <= tol."""
    if len(a) == 0:
        return []
    out, s, p = [], a[0], a[0]
    for v in a[1:]:
        if v - p > tol:
            out.append((int(s), int(p)))
            s = v
        p = v
    out.append((int(s), int(p)))
    return out


def dilate1d(v: np.ndarray, k: int) -> np.ndarray:
    """Binary dilation of a 1-D boolean vector by k samples each way."""
    out = v.copy()
    for s in range(1, k + 1):
        out[:-s] |= v[s:]
        out[s:] |= v[:-s]
    return out


def _erase_slivers(line: np.ndarray, min_len: int) -> None:
    """Zero, in place, every run of ink at least `min_len` long."""
    idx = np.where(line)[0]
    for a, b in runs1d(idx, tol=1):
        if b - a + 1 >= min_len:
            line[a:b + 1] = 0


def strip_grid(ink: np.ndarray, V, H_, bridge: int = 5, sliver: int = 0, chunk: int = 64,
               edge: float = 0.15, reach: int = 2) -> np.ndarray:
    """Delete the lattice, then SURGICALLY repair the curves it was crossing.

    Only the grid line's own band is erased (frame detection measures it to
    the pixel, antialiased edge rows included). Erasing a wider margin would
    also wipe out any drawn line running along the grid at a shallow angle --
    a 2 % pavement surface near a round elevation loses ~15 ft that way.
    The band is taken stretch by stretch (`chunk` px along the line): a row
    beside it, up to `reach` px out, inked over `edge` of a stretch is the
    line's own edge there. A blurred, resampled or straightened scan softens
    a line and lets it drift a pixel along its length; its edges fall either
    side of the ink threshold in fragments which, left standing, get bridged
    back across the band or chained into a line of their own.

    Zeroing a band also punches a hole through every curve that CROSSES it,
    so each band is re-bridged where ink exists on both sides within
    +-`bridge` px: continuity is restored without restoring the grid."""
    m = ink.copy()
    lines = [(m.T, s_, e) for s_, e in V] + [(m, s_, e) for s_, e in H_]      # each band as rows of a view
    spans = []
    for view, s_, e in lines:
        spans.append(_band_spans(view, s_, e, chunk, edge, reach))
        for c0, c1, a, b in spans[-1]:
            view[a:b + 1, c0:c1] = 0
    if sliver:
        # a skewed, uneven or heavier-than-detected grid line leaves long
        # straight slivers beside its band; data lines only cross there
        # briefly, so long runs are grid
        near = 4
        for view, s_, e in lines:
            for r in list(range(s_ - near, s_)) + list(range(e + 1, e + 1 + near)):
                if 0 <= r < view.shape[0]:
                    row = view[r].copy()
                    _erase_slivers(row, sliver)
                    view[r] = row
    for (view, _, _), sp in zip(lines, spans):
        for c0, c1, a, b in sp:
            if a - 1 < 0 or b + 1 >= view.shape[0]:
                continue
            both = _bridged(view[a - 1, c0:c1].astype(bool), view[b + 1, c0:c1].astype(bool), bridge)
            view[a:b + 1, c0:c1] |= both[None, :].astype(np.uint8)
    return m


def _band_spans(view: np.ndarray, s: int, e: int, chunk: int, edge: float, reach: int, along: float = 0.35):
    """A grid band (rows s..e of `view`) stretch by stretch: (c0, c1, first
    row, last row), grown by the rows beside it inked over `edge` of the
    stretch, up to `reach` rows out -- on a side where the line is soft: its
    edge shows in `along` of the stretches, all along the line. Beside a
    sharp line a row is inked only where a drawn line runs close by (a
    roadbed at a round elevation), and keeps it."""
    H, W = view.shape
    a0, b0 = max(0, s), min(H - 1, e)
    starts = range(0, W, chunk)

    def soft(r):
        return 0 <= r < H and np.mean([view[r, c0:c0 + chunk].mean() >= edge for c0 in starts]) >= along

    up = next((k - 1 for k in range(1, reach + 1) if not soft(a0 - k)), reach)
    down = next((k - 1 for k in range(1, reach + 1) if not soft(b0 + k)), reach)
    if not up and not down:
        return [(0, W, a0, b0)]
    out = []
    for c0 in starts:
        c1 = min(W, c0 + chunk)
        a, b = a0, b0
        for _ in range(up):
            if a > 0 and view[a - 1, c0:c1].mean() >= edge:
                a -= 1
        for _ in range(down):
            if b < H - 1 and view[b + 1, c0:c1].mean() >= edge:
                b += 1
        out.append((c0, c1, a, b))
    return out


def _bridged(a: np.ndarray, b: np.ndarray, bridge: int) -> np.ndarray:
    """Where to carry ink across an erased grid band, given the ink on its
    two sides. Each line crossing the band is a run of ink on either side;
    a run is joined to the run it touches on the other side, or else to its
    nearest one there within `bridge` when that one's nearest is it in turn,
    across their overlap, or the gap between them. Filling everything near
    ink on both sides instead would also fill the span between two lines
    crossing the band close together (ground passing by the template) and
    weld them into one stroke."""
    out = np.zeros_like(a)
    ra, rb = np.array(runs1d(np.flatnonzero(a), tol=1)), np.array(runs1d(np.flatnonzero(b), tol=1))
    if not len(ra) or not len(rb):
        return out
    hi_start = np.maximum(ra[:, None, 0], rb[None, :, 0])
    lo_end = np.minimum(ra[:, None, 1], rb[None, :, 1])
    gap = hi_start - lo_end - 1                 # < 0: the runs overlap
    na, nb = gap.argmin(axis=1), gap.argmin(axis=0)
    join = gap <= 0
    i = np.arange(len(ra))
    mutual = (nb[na] == i) & (gap[i, na] <= bridge)
    join[i[mutual], na[mutual]] = True
    for i, j in zip(*np.nonzero(join)):
        lo, hi = sorted((hi_start[i, j], lo_end[i, j]))
        out[lo:hi + 1] = True
    return out


def denoise(roi: np.ndarray, up: float = 1.0) -> tuple[np.ndarray, int]:
    """Drop annotation glyphs, traffic arrows and survey specks. Discriminator is
    BOX FILL RATIO: a digit fills ~0.3-0.6 of its bounding box, a 2 px line
    through the same box fills < 0.15. Builds one keep/drop table indexed by
    component label and applies it in a single vectorized pass. `up`:
    canonical px per scanned px -- lettering and pen strokes have a size on
    paper, so on a sheet drawn at a small scale (1 in = 40 ft), enlarged to
    the canonical resolution, they come out that much bigger."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(roi, 8)
    drop = np.zeros(n, dtype=bool)
    dropped = 0
    for k in range(1, n):
        x, y, w, h, a = st[k]
        f = a / (w * h)
        # glyph-sized and dense | a solid symbol (filled arrow, north arrow) --
        # compact, never a long flat box like a pavement section | a speck
        if (8 * up <= w <= 70 * up and 10 * up <= h <= 55 * up and f >= 0.30) \
           or (a > 700 * up * up and f >= 0.35 and max(w, h) <= 4 * min(w, h)) \
           or (a <= 12 * up * up and w <= 7 * up and h <= 7 * up):
            drop[k] = True
            dropped += 1
    keep = np.where(drop[lab], 0, roi).astype(roi.dtype)
    return keep, dropped


def split_dash_solid(mask: np.ndarray, up: float = 1.0):
    """Existing ground is DASHED, the proposed template is SOLID. Estimate the
    dash length from the component-width distribution and cut at 3x the median,
    so the threshold adapts to scan resolution instead of being hard-coded.

    A short component only counts as a ground DASH when it is shaped like a
    piece of line: thin, elongated and not steep. Glyph fragments of labels
    ("0.80% MATCH EXIST", rotated elevation callouts) are compact or upright
    and would otherwise pull the ground trace up into the annotation text.

    Returns (dash, solid, threshold, typical dash width)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    ws = st[1:, cv2.CC_STAT_WIDTH] if n > 1 else np.array([])
    dash_w = float(np.median(ws)) if len(ws) else 12.0
    thr = max(40.0, 3.0 * dash_w)
    is_solid = np.zeros(n, dtype=bool)
    is_dash = np.zeros(n, dtype=bool)
    if n > 1:
        hs = st[1:, cv2.CC_STAT_HEIGHT]
        is_solid[1:] = np.maximum(ws, hs) >= thr
        for k in np.where(~is_solid[1:])[0] + 1:
            w, h, area = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT], st[k, cv2.CC_STAT_AREA]
            length = float(np.hypot(w, h))
            thickness = area / max(length, 1.0)
            is_dash[k] = (length >= 6 * up and thickness <= 7.0 * up and length / max(thickness, 1.0) >= 1.8
                          and h <= 1.6 * w + 3)
    is_dash &= _collinear_support(lab, st, is_dash, reach=24.0 * up)
    solid = np.where(is_solid[lab], mask, 0).astype(mask.dtype)
    dash = np.where(is_dash[lab], mask, 0).astype(mask.dtype)
    return dash, solid, thr, dash_w


def _collinear_support(lab: np.ndarray, st: np.ndarray, is_dash: np.ndarray,
                       reach: float = 24.0, max_turn_deg: float = 25.0) -> np.ndarray:
    """A dash of a dashed line is continued by another dash: one whose end
    lies within `reach` px, roughly along the same direction. Strokes of
    label text -- a rotated "1", the bar of a "7", a hyphen -- are short
    lines too, but their neighbours sit beside them, not in line with them."""
    ids = np.where(is_dash)[0]
    keep = np.zeros(len(is_dash), bool)
    if len(ids) < 2:
        return keep
    cen, dirs, half = [], [], []
    for k in ids:
        x, y, w, h = st[k, :4]
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == k)
        pts = np.stack([xs + x, ys + y], 1).astype(float)
        c = pts.mean(0)
        if len(pts) > 2:
            ev, evec = np.linalg.eigh(np.cov((pts - c).T))
            d = evec[:, 1]
        else:
            d = np.array([1.0, 0.0])
        if d[0] < 0:
            d = -d
        cen.append(c)
        dirs.append(d)
        half.append(max(1.0, float(np.abs((pts - c) @ d).max())))
    cen, dirs, half = np.array(cen), np.array(dirs), np.array(half)
    cos_turn = np.cos(np.radians(max_turn_deg))
    for i in range(len(ids)):
        v = cen - cen[i]
        dist = np.hypot(v[:, 0], v[:, 1])
        gap = dist - half - half[i]                       # end-to-end distance
        ok = (gap <= reach) & (dist > 0)
        if not ok.any():
            continue
        u = v[ok] / dist[ok][:, None]
        along = np.abs(u @ dirs[i]) >= cos_turn           # neighbour lies in line
        parallel = np.abs(dirs[ok] @ dirs[i]) >= cos_turn  # and points the same way
        if (along & parallel).any():
            keep[ids[i]] = True
    return keep


def candidates(mask: np.ndarray, max_run: int = 30) -> list[list[float]]:
    """Per column, the mid-point of every vertical ink run = a curve observation."""
    out = []
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x])[0]
        c = []
        for s, e in runs1d(col, tol=2):
            if e - s + 1 <= max_run:
                c.append((s + e) / 2.0)
            else:
                c += [float(s), float(e)]
        out.append(c)
    return out


def best_path(cands, reward=3.0, win=260, slope=0.6, slack=8.0, gap_pen=0.04,
              weights=None) -> dict[int, float]:
    """Max-scoring chain through the candidate points (Viterbi over a DAG).

    A greedy left-to-right tracker breaks at the first long dash gap and then
    coasts away from the curve. Scoring globally -- +reward per point used,
    -|dy| for roughness, -gap_pen per skipped column -- recovers the whole
    surface including the stretch hidden behind the proposed template.

    Vectorized over the sliding window: the original was a pure-Python
    `for i: for j in range(start, i)` double loop, which is fine for a
    handful of candidate points but scales very badly on a wide, ink-dense
    real drawing with thousands of them -- confirmed as the dominant cost
    of the whole per-region pipeline on real GDOT plan sets (~5s/region,
    the large majority of total processing time). This computes the same
    scores with numpy array ops over each window instead of a Python-level
    inner loop. Tie-breaking matches exactly: np.argmax returns the FIRST
    occurrence of the max value for a tie, same as the original's strict
    `s > score[i]` comparison while iterating j in increasing order.

    `weights` (parallel to `cands`) scales the reward of individual points,
    so weaker evidence can be offered to the tracer without outranking
    strong evidence."""
    if weights is None:
        weights = [[1.0] * len(c) for c in cands]
    pts = [(x, y, w) for x, (c, cw) in enumerate(zip(cands, weights)) for y, w in zip(c, cw)]
    if not pts:
        return {}
    n = len(pts)
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    rw = reward * np.array([p[2] for p in pts], dtype=float)
    score = rw.copy()
    prev = np.full(n, -1)
    start = 0
    for i in range(n):
        xi, yi = xs[i], ys[i]
        while xs[start] < xi - win:
            start += 1
        if start >= i:
            continue
        wx = xs[start:i]
        wy = ys[start:i]
        dx = xi - wx
        dy = np.abs(yi - wy)
        valid = (dx > 0) & (dy <= slope * dx + slack)
        if not valid.any():
            continue
        s = score[start:i] + rw[i] - dy - gap_pen * (dx - 1)
        s = np.where(valid, s, -np.inf)
        j_local = int(np.argmax(s))
        if s[j_local] > score[i]:
            score[i] = s[j_local]
            prev[i] = start + j_local
    k = int(np.argmax(score))
    path = []
    while k >= 0:
        path.append(k)
        k = prev[k]
    return {int(xs[i]): float(ys[i]) for i in reversed(path)}


def square_steps(chain: dict[int, float], mask: np.ndarray, min_step: float = 4.0, cover: float = 0.7,
                 near: float = 3.0) -> dict[int, float]:
    """Vertical faces as drawn: square corners, not slopes. A chain moves at
    most `slope` px per column (best_path), so where the line drops down a
    wall face, a curb or any vertical step, it skips the columns after the
    face and densify() bridges them with a slope -- cutting a triangle off
    the corner at the face's foot (a 3 ft wall face loses ~4 ft of floor).
    Where consecutive points more than a column apart differ by `min_step`
    px or more and a column between them is inked over `cover` of the rise
    (the face), the columns before the face follow the ink at the first
    point's level, those after it the ink at the second's (within `near`
    px, the level carried flat where there is none)."""
    if len(chain) < 2:
        return chain
    xs = sorted(chain)
    out = dict(chain)
    H = mask.shape[0]
    for a, b in zip(xs, xs[1:]):
        ya, yb = chain[a], chain[b]
        if b - a < 2 or abs(yb - ya) < min_step:
            continue
        lo, hi = int(np.ceil(min(ya, yb) + 2)), int(np.floor(max(ya, yb) - 2))
        if hi <= lo:
            continue
        faces = [x for x in range(a, b + 1) if mask[max(0, lo):min(H, hi + 1), x].mean() >= cover]
        if not faces:
            continue
        f = faces[len(faces) // 2]
        for x in range(a + 1, b):
            out[x] = _stroke_near(mask, x, ya if x < f else yb, near)
    return out


def _stroke_near(mask: np.ndarray, x: int, y: float, near: float) -> float:
    """Centre of the stroke in column x nearest row y, within `near` px; y if none."""
    r0 = max(0, int(y - near - 2))
    runs = runs1d(np.flatnonzero(mask[r0:int(y + near + 3), x]) + r0, tol=1)
    mids = [(s_ + e) / 2.0 for s_, e in runs if abs((s_ + e) / 2.0 - y) <= near]
    return float(min(mids, key=lambda m: abs(m - y))) if mids else float(y)


def snap_top(mask: np.ndarray, chain: dict[int, float], rad: int = 9) -> dict[int, float]:
    """The template is drawn as a thin box (surface + pavement structure), so the
    finished surface is the TOPMOST stroke -- and we want that stroke's CENTRE.
    Taking its upper edge biases the whole surface up by half a stroke width,
    which shows up directly as a fill/cut error."""
    out = {}
    for x, y in chain.items():
        lo, hi = max(0, int(y) - rad), min(mask.shape[0], int(y) + rad + 1)
        w = np.where(mask[lo:hi, x])[0]
        if len(w) == 0:
            out[x] = y
            continue
        a, b = runs1d(w, tol=1)[0]
        out[x] = float(lo + (a + b) / 2.0)
    return out


def snap_bottom(mask: np.ndarray, chain: dict[int, float], rad: int) -> dict[int, float]:
    """Subgrade under a traced finished-grade line: the LOWEST template stroke
    within `rad` px below it -- the bottom of the pavement box (or curb)
    where there is one, the line itself where there isn't (shoulders, side
    slopes, ditches are drawn as a single line). A stroke that runs on past
    `rad` -- a guardrail post, a wall face -- is not pavement and ends the
    search."""
    out = {}
    H = mask.shape[0]
    for x, y in chain.items():
        y0 = int(round(y))
        lo, hi = max(0, y0 - 2), min(H, y0 + rad + 1)
        col = np.where(mask[lo:hi, x])[0]
        best = y
        for a, b in runs1d(col + lo, tol=1):
            if b < y0 - 2:
                continue
            if b >= hi - 1:            # runs on beyond reach
                break
            best = (a + b) / 2.0
        out[x] = float(best)
    return out


def despike(y: np.ndarray, win: int, thresh_px: float) -> tuple[np.ndarray, int]:
    """Reject outliers only -- do NOT smooth. Leader lines, witness ticks and
    dimension arrows that survived the glyph filter show up as isolated jumps;
    genuine break-points (ditch invert, catch point, shoulder hinge) must come
    through untouched, so flagged samples are replaced by interpolation from
    their good neighbours rather than by a filtered value."""
    ok = ~np.isnan(y)
    idx = np.where(ok)[0]
    if len(idx) < win:
        return y, 0
    v = y[idx].copy()
    bad = np.abs(v - median_filter(v, size=win, mode="nearest")) > thresh_px
    if bad.any() and (~bad).sum() >= 2:
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    out = y.copy()
    out[idx] = v
    return out, int(bad.sum())


def trim_buried_tail(x: np.ndarray, gpx: np.ndarray, dpx: np.ndarray, ppf: float,
                      run_ft: float, depth_ft: float, end_ft: float = 1.0,
                     roadway_depth_ft: float = 15.0, subgrade: np.ndarray | None = None):
    """Temporary pipes and other buried services run on past the template's toe.
    They read as a LONG, DEEP run of 'proposed below existing' that reaches the
    end of the chain and STOPS THERE, still inside the ground. A graded cut
    can be just as long and deep, but its back slope climbs out and ends on
    the ground at the catch point -- so a run is only trimmed when the chain
    ends more than `end_ft` below existing ground. A run that passes under
    the middle of the section (x = 0, the roadway) AND lies deeper than
    `roadway_depth_ft` is the template itself -- a deep cut that the frame
    cuts off before its back slope reaches the ground (SR 5, 18-45 ft); pipes
    crossing under the road on the labelled sheets lie 8-12 ft down and are
    still trimmed. A run is one continuous stretch of design: a widening
    drawn apart from the rest of the template is judged on its own. (A template ending at a wall's foot is first carried up
    the back slope by climb_from_wall, so it no longer ends below ground.)
    A pipe is one line; a run that ends as a pavement box (`subgrade` well
    below the surface) a few feet under existing ground is the new road built
    down through the old one, up to where the drawing stops -- cut, kept."""
    ov = ~np.isnan(gpx) & ~np.isnan(dpx)
    if ov.sum() < 10:
        return dpx, []
    c = np.where(ov)[0]
    xx = x[c]
    d = (gpx[c] - dpx[c]) / ppf
    sign = np.sign(np.where(np.abs(d) < 0.10, 0, d))
    gap_px = 3.0 * ppf                  # a run ends where the design does (separate template pieces)
    runs, i = [], 0
    while i < len(d):
        if sign[i] < 0:
            j = i + 1
            while j < len(d) and sign[j] <= 0 and c[j] - c[j - 1] <= gap_px:
                j += 1
            runs.append((i, j - 1, xx[j - 1] - xx[i], float(-d[i:j].min())))
            i = j
        else:
            i += 1
    out = dpx.copy()
    killed = []
    for a, b, L, dep in runs:
        if run_ft > L or dep < depth_ft or (xx[a] < 0 < xx[b] and dep > roadway_depth_ft):
            continue
        if xx[-1] - xx[b] <= 2.0 and -d[-1] > end_ft:
            if subgrade is not None and _ends_as_pavement(c, d, b, -1, dpx, subgrade, ppf):
                continue
            out[c[a]:] = np.nan
            killed.append(dict(side="right", x_from=round(float(xx[a]), 2),
                                x_to=round(float(xx[-1]), 2),
                                run_ft=round(float(L), 1), depth_ft=round(dep, 2)))
        elif xx[a] - xx[0] <= 2.0 and -d[0] > end_ft:
            if subgrade is not None and _ends_as_pavement(c, d, a, 1, dpx, subgrade, ppf):
                continue
            out[:c[b] + 1] = np.nan
            killed.append(dict(side="left", x_from=round(float(xx[0]), 2),
                                x_to=round(float(xx[b]), 2),
                                run_ft=round(float(L), 1), depth_ft=round(dep, 2)))
    return out, killed


def _ends_as_pavement(c, d, end: int, step: int, dpx: np.ndarray, bpx: np.ndarray, ppf: float,
                      tail_ft: float = 25.0, max_depth_ft: float = 5.0, box_ft: float = 0.3) -> bool:
    """The run's end inside the ground (index `end`, the run lying in
    direction `step` from it) is a pavement box within max_depth_ft of the
    ground: the surface line has a subgrade line under it along most of the
    last tail_ft."""
    n = int(tail_ft * ppf)
    lo, hi = (end, end + n) if step > 0 else (end - n + 1, end + 1)
    lo, hi = max(lo, 0), min(hi, len(c))
    if hi - lo < 0.5 * n:
        return False
    cols = c[lo:hi]
    box = (bpx[cols] - dpx[cols]) / ppf >= box_ft
    return bool(box.mean() >= 0.6 and np.median(-d[lo:hi]) <= max_depth_ft)


def climb_from_wall(D: np.ndarray, G: np.ndarray, mask: np.ndarray, ppf: float,
                    min_drop_ft: float = 3.0, max_run_ft: float = 150.0) -> tuple[np.ndarray, list[dict]]:
    """A cut retained by a wall: the traced template ends at the wall's FOOT,
    deep below ground, because the wall face is vertical (one column cannot
    hold two rows of a single-valued line). Above the wall the back slope
    climbs on to meet the ground, drawn as its own line. From each end of the
    design that stops more than `min_drop_ft` below ground, look for ink
    starting between that end and the ground and follow it outward while it
    keeps climbing; if it reaches the ground, it is the rest of the template.
    Returns (design, [{side, x_from, x_to}] in columns)."""
    W = len(D)
    ok = np.where(~np.isnan(D))[0]
    if len(ok) < 2:
        return D, []
    out, notes = D.copy(), []
    cols = {}

    def cands(x):
        if x not in cols:
            col = np.where(mask[:, x])[0]
            cols[x] = [float(s_) for s_, _ in runs1d(col, tol=1)] + [float(e) for _, e in runs1d(col, tol=1)]
        return cols[x]

    for e, step in ((int(ok[0]), -1), (int(ok[-1]), 1)):
        if np.isnan(G[e]) or (D[e] - G[e]) / ppf < min_drop_ft:
            continue
        lo, hi = G[e] + ppf, D[e] - 0.5 * ppf          # rows strictly between ground and the wall foot
        seeds = sorted({c for k in range(0, int(1.5 * ppf)) for x in [e + step * k] if 0 <= x < W
                        for c in cands(x) if lo < c < hi}, reverse=True)
        for seed in seeds[:6]:
            pts, prev, x, miss = [], seed, e, 0
            while 0 <= x + step < W and abs(x - e) < max_run_ft * ppf:
                x += step
                near = [c for c in cands(x) if prev - (2.5 + 2.5 * miss) <= c <= prev + 1.0]
                if not near:
                    miss += 1
                    if miss > ppf:                     # 1 ft with no ink: the line has ended
                        break
                    continue
                prev, miss = min(near, key=lambda c: abs(c - prev)), 0
                pts.append((x, prev))
                if not np.isnan(G[x]) and prev <= G[x] + 1.5:
                    break
            else:
                continue
            if not pts or np.isnan(G[pts[-1][0]]) or pts[-1][1] > G[pts[-1][0]] + 1.5:
                continue                               # never got back to the ground
            if abs(pts[-1][0] - e) < 2 * ppf or (seed - pts[-1][1]) < 2 * ppf:
                continue                               # too short to be a slope
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            order = np.argsort(xs)
            span = np.arange(min(xs.min(), e + step), max(xs.max(), e + step) + 1)
            span = span[(span >= 0) & (span < W)]
            out[span] = np.interp(span, xs[order], ys[order])
            notes.append(dict(side="left" if step < 0 else "right", x_from=int(span.min()), x_to=int(span.max())))
            break
    return out, notes


def refine_gaps(chain: dict[int, float], full_mask: np.ndarray, ppf: float,
                 min_gap_ft: float, tol_px: float):
    """Inside a long interpolation gap the surface's own dashes usually still
    exist -- they were simply absorbed into a solid component they touch, so the
    dash/solid split never offered them. Re-hunt for ink near the interpolated
    path in the FULL mask and accept it when it is close enough to be the same
    surface. This is a recall pass, not a smoother: it only ever adds points."""
    if len(chain) < 2:
        return chain, 0
    xs = np.array(sorted(chain))
    ys = np.array([chain[x] for x in xs])
    add = {}
    for i in np.where(np.diff(xs) / ppf > min_gap_ft)[0]:
        a, b, ya, yb = xs[i], xs[i + 1], ys[i], ys[i + 1]
        for x in range(a + 1, b):
            yp = ya + (yb - ya) * (x - a) / (b - a)
            col = np.where(full_mask[:, x])[0]
            if not len(col):
                continue
            best = min(((s_ + e) / 2 for s_, e in runs1d(col, tol=2)), key=lambda c: abs(c - yp))
            if abs(best - yp) <= tol_px:
                add[x] = best
    out = dict(chain)
    out.update(add)
    return out, len(add)


def interp_spans(chain: dict[int, float], W: int, ppf: float, warn_ft: float):
    """Boolean per column: True where the surface is only a straight-line guess
    across a gap wider than `warn_ft`. Area computed there is not supported by
    ink and is reported separately rather than quietly folded into the total."""
    m = np.zeros(W, bool)
    if len(chain) < 2:
        return m, 0.0
    xs = np.array(sorted(chain))
    d = np.diff(xs) / ppf
    for i in np.where(d > warn_ft)[0]:
        m[xs[i]:xs[i + 1] + 1] = True
    return m, float(d.max()) if len(d) else 0.0


def densify(chain: dict[int, float], W: int) -> np.ndarray:
    """Chain -> dense per-column array, linearly interpolated, NaN outside."""
    g = np.full(W, np.nan)
    if not chain:
        return g
    xs = np.array(sorted(chain))
    ys = np.array([chain[x] for x in xs])
    grid = np.arange(W)
    m = (grid >= xs[0]) & (grid <= xs[-1])
    g[m] = np.interp(grid[m], xs, ys)
    return g


def split_chain(chain: dict[int, float], max_gap: int) -> list[dict[int, float]]:
    """Break a traced chain wherever consecutive points are more than
    `max_gap` columns apart."""
    if not chain:
        return []
    xs = sorted(chain)
    pieces, cur = [], {xs[0]: chain[xs[0]]}
    for a, b in zip(xs, xs[1:]):
        if b - a > max_gap:
            pieces.append(cur)
            cur = {}
        cur[b] = chain[b]
    pieces.append(cur)
    return pieces


def drop_structures(pieces: list[dict[int, float]], ground: np.ndarray, ppf: float,
                    tie_ft: float = 2.0, min_len_ft: float = 2.0):
    """Keep the pieces of the design line that are earthwork.

    A roadway template always meets existing ground through its side slopes
    (or a wall). A bridge deck, a culvert top or a sign structure is drawn
    with the same solid pen but hangs ABOVE the ground at both of its ends.
    Measuring "fill" under it would count air as embankment, so such pieces
    are dropped and reported. Slivers too short to be a surface are dropped
    silently. Returns (kept pieces, notes)."""
    kept, notes = [], []
    for pc in pieces:
        xs = sorted(pc)
        if (xs[-1] - xs[0]) / ppf < min_len_ft:
            continue
        floating = 0
        for x in (xs[0], xs[-1]):
            g = ground[x] if 0 <= x < len(ground) else np.nan
            if not np.isnan(g) and (g - pc[x]) / ppf > tie_ft:   # design above ground, untied
                floating += 1
        if floating == 2:
            notes.append(dict(x_from=xs[0], x_to=xs[-1]))
            continue
        kept.append(pc)
    return kept, notes


def fill_along_gridlines(chain: dict[int, float], grid_rows: list[float], min_gap: int,
                         tol: float = 9.0) -> dict[int, float]:
    """A drawn line running along (or crossing at a shallow angle) a grid
    line is erased together with the grid -- a pavement surface sitting at a
    round elevation such as 660.00 is the common case. When a gap in the
    traced chain starts and ends close to the same grid row, the gap was
    made by grid removal, not by the drawing: bridge it straight."""
    if len(chain) < 2 or not grid_rows:
        return chain
    rows = np.asarray(grid_rows, float)
    xs = sorted(chain)
    out = dict(chain)
    for a, b in zip(xs, xs[1:]):
        if b - a <= min_gap:
            continue
        ya, yb = chain[a], chain[b]
        r = rows[np.argmin(np.abs(rows - (ya + yb) / 2))]
        if abs(ya - r) <= tol and abs(yb - r) <= tol:
            for x in range(a + 1, b):
                out[x] = ya + (yb - ya) * (x - a) / (b - a)
    return out


def template_strokes(solid: np.ndarray, link_px: int, min_px: int, grid_rows: list[float] | None = None,
                     grid_tol: float = 9.0, grid_gap: int = 0) -> np.ndarray:
    """Keep the solid strokes that make up the roadway template.

    The template (pavement, shoulders, side slopes, ditches) is one network
    of strokes that touch or nearly touch. Leader lines, utility callouts and
    label underlines are drawn with the same solid pen but stand apart from
    it, and they are short. Group strokes lying within `link_px` of each
    other and keep only the wide groups: the widest one, plus any other spanning at
    least 30 % of it (a second roadway beyond a median, the far side of a
    structure) and at least `min_px`.

    On a scan, a stretch of template lying on a grid line (a sidewalk or
    shoulder at a round elevation) is erased with the grid, leaving the
    strokes either side further apart than `link_px`: strokes meeting the
    same grid row (`grid_rows`, within `grid_tol` px) up to `grid_gap` px
    apart are grouped as well."""
    if not solid.any():
        return solid
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (link_px, link_px))
    linked = cv2.dilate(solid, k)
    H = solid.shape[0]
    for r in grid_rows or []:
        band = solid[max(0, int(r - grid_tol)):min(H, int(r + grid_tol) + 1)]
        # strokes running into the row, not specks or uprights crossing it
        runs = [(a, b) for a, b in runs1d(np.flatnonzero(band.any(axis=0)), tol=1) if b - a + 1 >= grid_tol]
        for (_, e), (s_, _) in zip(runs, runs[1:]):
            if link_px < s_ - e <= grid_gap:
                linked[int(r), e:s_ + 1] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(linked, 8)
    if n <= 2:
        return solid
    widths = st[1:, cv2.CC_STAT_WIDTH] - link_px
    cut = max(0.3 * widths.max(), min_px)
    keep = np.zeros(n, bool)
    keep[1:] = widths >= cut
    if not keep.any():
        keep[1 + int(np.argmax(widths))] = True
    _continue_template(lab, st, keep, link_px)
    return np.where(keep[lab], solid, 0).astype(solid.dtype)


def _ends(lab: np.ndarray, st, k: int) -> tuple[tuple[int, float], tuple[int, float]]:
    """A group's leftmost and rightmost ink: (column, mean row) at each end."""
    x0, x1 = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] - 1
    y0, y1 = st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]
    band = lab[y0:y1]
    row = lambda x: float(np.mean(np.where(band[:, x] == k)[0])) + y0
    return (x0, row(x0)), (x1, row(x1))


def _continue_template(lab: np.ndarray, st, keep: np.ndarray, link_px: int) -> None:
    """Keep short stroke groups that carry the template on: one starting
    where a kept group ends -- a little beyond its last ink (up to 1.75 x
    link_px further out) and at the same height (within a quarter of
    link_px) -- like a side slope drawn after a small break in the linework.
    Text underlines and leaders sit above or below the template, so they
    stay out; so does anything climbing steeper than 2:1 from the break (a
    leader rising from the template's end to a note). Repeats until nothing
    more joins."""
    n = len(keep)
    far, dy = 1.75 * link_px, 0.25 * link_px
    ends = {k: _ends(lab, st, k) for k in range(1, n)}
    changed = True
    while changed:
        changed = False
        kept = [k for k in range(1, n) if keep[k]]
        for k in range(1, n):
            if keep[k]:
                continue
            (lx, ly), (rx, ry) = ends[k]
            if st[k, cv2.CC_STAT_HEIGHT] - link_px > 0.5 * (st[k, cv2.CC_STAT_WIDTH] - link_px):
                continue
            for t in kept:
                (tlx, tly), (trx, try_) = ends[t]
                if (0 < lx - trx <= far and abs(ly - try_) <= dy) or (0 < tlx - rx <= far and abs(ry - tly) <= dy):
                    keep[k] = changed = True
                    break


def drop_layer_lines(dash: np.ndarray, template: np.ndarray, link: int = 29, min_w: int = 72,
                     min_off: int = 6, max_off: int = 43, max_std: float = 3.0,
                     max_slope: float = 0.10) -> np.ndarray:
    """Remove dashed runs that are existing pavement layers, not ground.

    An existing road is drawn as a dashed surface plus a dashed line for the
    bottom of its pavement. Where the new template sits on the old road the
    upper (ground) line hides under the template's solid ink and only the
    layer line still shows as dashes -- so the tracer would dive onto it. The
    tell: it runs entirely beneath the template, parallel to it at a
    constant offset, and it starts or ends with a step down from ground that
    carries on above it. (Ground genuinely parallel to the new grade, e.g.
    under a thin raise, joins the ground beside it without a step and stays.)
    Defaults are canonical pixels (14.4 px/ft): link 2 ft, min width 5 ft,
    offset 0.4--3 ft, offset scatter <= 0.2 ft; pavement cross slope <= 10 %."""
    if not dash.any() or not template.any():
        return dash
    joined = cv2.dilate(dash, cv2.getStructuringElement(cv2.MORPH_RECT, (link, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, 8)
    out = dash.copy()
    for i in range(1, n):
        x0, w = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_WIDTH]
        if w < min_w:
            continue
        offs, cols, pts = [], 0, []
        for x in range(x0, x0 + w):
            ys = np.where((lab[:, x] == i) & (dash[:, x] > 0))[0]
            if len(ys) == 0:
                continue
            cols += 1
            y = int(ys.mean())
            pts.append((x, y))
            above = np.where(template[max(0, y - max_off):max(0, y - min_off + 1), x])[0]
            if len(above):
                offs.append(y - (max(0, y - max_off) + above.max()))
        if cols == 0 or len(offs) < 0.9 * cols:
            continue
        px, py = np.array(pts, float).T
        if abs(np.polyfit(px, py, 1)[0]) > max_slope:     # a side slope, not a pavement layer
            continue
        if np.std(offs) <= max_std and _steps_off_ground(dash, lab, i, x0, w, min_off, max_off, link):
            out[(lab == i) & (dash > 0)] = 0
    return out


def _steps_off_ground(dash, lab, i, x0, w, min_step, max_rise, reach) -> bool:
    """Does dashed run `i` begin or end with a vertical step down from other
    dashes just beyond it -- ground carrying on ABOVE it (by min_step to
    max_rise)? Real ground is continuous; a layer line branches off below it."""
    own = (lab[:, x0:x0 + w] == i) & (dash[:, x0:x0 + w] > 0)
    cols = np.where(own.any(axis=0))[0]
    if len(cols) == 0:
        return False
    first, last = x0 + cols[0], x0 + cols[-1]        # real dash ends, not the link padding
    for end, outside in ((first, range(max(0, first - reach), first)),
                         (last, range(last + 1, min(dash.shape[1], last + 1 + reach)))):
        y = np.where(own[:, end - x0])[0].mean()
        for x in outside:
            other = np.where((dash[:, x] > 0) & (lab[:, x] != i))[0]
            rise = y - other
            if ((rise >= min_step) & (rise <= max_rise)).any():
                return True
    return False


def ground_candidates(dash: np.ndarray, template: np.ndarray, hidden_weight: float):
    """Candidate points for tracing existing ground: its own dashes, plus the
    template's strokes at a fraction of the reward. Dashed ground that slips
    behind the solid template (where the two coincide, e.g. widening onto
    existing pavement) leaves no dashes of its own; following the template
    there beats bridging blind in a straight line, while anywhere real dashes
    exist they win."""
    cd, ct = candidates(drop_layer_lines(dash, template)), candidates(template)
    cands = [a + b for a, b in zip(cd, ct)]
    weights = [[1.0] * len(a) + [hidden_weight] * len(b) for a, b in zip(cd, ct)]
    return cands, weights


def trace_ground(dash: np.ndarray, template: np.ndarray, hidden_weight: float, win: int, ppf: float,
                 min_run_ft: float = 15.0, min_dash_frac: float = 0.1) -> dict[int, float]:
    """Existing ground through its dashes, borrowing the template where the
    ground hides behind it (ground_candidates). But under a tall embankment
    or in a deep cut the ground is far from the template, drawn as faint,
    broken dashes: following the template's continuous strokes then scores
    better than hopping between them, and the section measures ~0 cut / fill.
    So where the trace rides the template for more than `min_run_ft` while
    at least `min_dash_frac` of those columns hold ground dashes elsewhere,
    the ground is not hidden there: trace those columns on the dashes alone."""
    cd, ct = candidates(drop_layer_lines(dash, template)), candidates(template)
    cands = [a + b for a, b in zip(cd, ct)]
    weights = [[1.0] * len(a) + [hidden_weight] * len(b) for a, b in zip(cd, ct)]
    path = best_path(cands, win=win, weights=weights)
    near = lambda y, ys: any(abs(y - v) <= 3 for v in ys)
    on_tpl = np.zeros(len(cd), bool)
    elsewhere = np.zeros(len(cd), bool)
    for x, y in path.items():
        if near(y, ct[x]) and not near(y, cd[x]):
            on_tpl[x] = True
    for x in range(len(cd)):
        elsewhere[x] = any(abs(v - w) > 0.5 * ppf for v in cd[x] for w in ct[x]) if ct[x] else bool(cd[x])
    # runs of columns where the trace rides the template (small gaps in the path allowed)
    xs = np.array(sorted(path))
    release = np.zeros(len(cd), bool)
    if len(xs):
        run_start = None
        for a, b in zip(xs, list(xs[1:]) + [None]):
            if on_tpl[a] and run_start is None:
                run_start = a
            if run_start is not None and (b is None or not on_tpl[b] or b - a > ppf):
                end = a
                if end - run_start >= min_run_ft * ppf:
                    span = slice(run_start, end + 1)
                    if elsewhere[span].mean() >= min_dash_frac:
                        release[span] = True
                run_start = None
    if not release.any():
        return path
    cands = [a if release[x] else a + b for x, (a, b) in enumerate(zip(cd, ct))]
    weights = [[1.0] * len(a) + ([] if release[x] else [hidden_weight] * len(b))
               for x, (a, b) in enumerate(zip(cd, ct))]
    return best_path(cands, win=win, weights=weights)


def drop_poles(solid: np.ndarray, ppf: float, min_ft: float = 8.0, max_w_ft: float = 1.5,
               clear_ft: float = 5.0, reach_ft: float = 20.0, wide_ft: float = 10.0) -> np.ndarray:
    """Remove utility poles (and other free-standing uprights) from the
    template mask. A pole is drawn in the template's solid pen -- a single
    line or a narrow box, often carried on below the ground, with a cross
    arm and maybe a down guy -- and where it stands in the graded section
    the tracers follow it: the design runs down the pole to its foot, the
    subgrade snaps to the box's floor or to a guy passing under a side slope,
    and the zone comes out notched at the pole.

    A pole is one or two vertical strokes at least `min_ft` tall within
    `max_w_ft` of each other, whose top stands clear: once its guys are set
    aside, no wide stroke (`wide_ft`, the template) comes within `clear_ft`
    of its top within `reach_ft` either side. A wall face is as tall, but
    its top meets the roadway (fill wall) or the back slope climbing on to
    the ground (cut wall), so it stays. The pole goes with its guys, every
    piece lying within its width (a box pole's top and floor) and every
    piece hanging off it that is no wider than a pole (a ground dash welded
    to its foot) or lies up by its top (the cross arm). Where the template
    crossed a pole or a guy, the gap left is a few pixels wide, bridged like
    any other."""
    if not solid.any():
        return solid
    vert = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(min_ft * ppf))))
    if not vert.any():
        return solid
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(vert, np.ones((1, 3), np.uint8)), 8)
    max_w = int(max_w_ft * ppf) + 3
    bars = sorted((k for k in range(1, n) if st[k, cv2.CC_STAT_WIDTH] <= max_w), key=lambda k: st[k, cv2.CC_STAT_LEFT])
    if not bars:
        return solid
    # the uprights side by side within a pole's width are one pole (a box)
    groups: list[list[int]] = []
    for k in bars:
        if groups and st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] - st[groups[-1][0], cv2.CC_STAT_LEFT] <= max_w:
            groups[-1].append(k)
        else:
            groups.append([k])
    H, W = solid.shape
    reach, clear = int(reach_ft * ppf), int(clear_ft * ppf)
    upright = np.isin(lab, bars)
    rest = (solid & ~upright).astype(np.uint8)
    poles = []
    for g in groups:
        x0 = min(st[k, cv2.CC_STAT_LEFT] for k in g)
        x1 = max(st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] for k in g)
        top = min(st[k, cv2.CC_STAT_TOP] for k in g)
        bot = max(st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] for k in g)
        poles.append((g, x0, x1, top, bot, _guys(rest, x0, x1, top, bot, reach, ppf)))
    guys = np.zeros_like(rest)
    for *_, gm in poles:
        guys |= gm
    m, lab2, st2, _ = cv2.connectedComponentsWithStats(rest & ~guys, 8)
    wide = np.zeros(m, bool)
    wide[1:] = st2[1:, cv2.CC_STAT_WIDTH] >= wide_ft * ppf
    wide_mask = wide[lab2]
    out = solid.copy()
    for g, x0, x1, top, _, gm in poles:
        if wide_mask[:min(H, top + clear + 1), max(0, x0 - reach):min(W, x1 + reach)].any():
            continue
        here = np.isin(lab, g)
        out[here] = 0
        out[gm > 0] = 0
        touching = np.zeros(m, bool)
        touching[np.unique(lab2[cv2.dilate((here | (gm > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])] = True
        inside = (st2[:, cv2.CC_STAT_LEFT] >= x0 - 1) & (st2[:, cv2.CC_STAT_LEFT] + st2[:, cv2.CC_STAT_WIDTH] <= x1 + 1)
        by_top = st2[:, cv2.CC_STAT_TOP] + st2[:, cv2.CC_STAT_HEIGHT] <= top + clear
        gone = inside | (touching & ((st2[:, cv2.CC_STAT_WIDTH] <= max_w) | by_top))
        gone[0] = False
        if gone.any():
            out[gone[lab2]] = 0
    return out


def _guys(rest: np.ndarray, x0: int, x1: int, top: int, bot: int, reach: int, ppf: float,
          min_ft: float = 6.0, steep: float = 1.0) -> np.ndarray:
    """Down guys of the pole standing in columns x0..x1 from row `top` to
    `bot`: straight strokes at least `min_ft` long, steeper than `steep`
    (rise over run; side slopes are flatter), whose upper end meets the
    upper part of the pole, followed on down their line to the pole's foot.
    Returns their pixels in `rest`."""
    out = np.zeros_like(rest)
    c0, c1 = max(0, x0 - reach), min(rest.shape[1], x1 + reach)
    r0, r1 = max(0, top - 4), min(rest.shape[0], bot + 4)
    win = rest[r0:r1, c0:c1]
    if not win.any():
        return out
    segs = cv2.HoughLinesP(win * 255, 1, np.pi / 180, int(3 * ppf), minLineLength=int(min_ft * ppf), maxLineGap=4)
    if segs is None:
        return out
    upper = top + 0.4 * (bot - top)
    band = np.zeros_like(win)
    for xa, ya, xb, yb in segs.reshape(-1, 4):
        if abs(yb - ya) < steep * abs(xb - xa) or xa == xb:
            continue                                         # not steep, or an upright (no guy)
        (hx, hy), (lx, ly) = ((xa, ya), (xb, yb)) if ya < yb else ((xb, yb), (xa, ya))
        # the guy's line, carried up, meets the pole's upper part -- maybe above
        # the frame, which can cut a pole's top off (where the guy runs into
        # the pole its detected end is a few px off)
        xc = (x0 + x1) / 2 - c0
        yc = hy + (xc - hx) * (ly - hy) / (lx - hx)
        if abs(hx - xc) <= 2 * ppf and hy + r0 <= upper and yc + r0 <= upper:
            # and on down to the anchor: where the guy crosses the template
            # the piece below is a separate, shorter stroke
            foot = win.shape[0] - 1
            ex = lx + (lx - hx) * (foot - ly) / (ly - hy)
            cv2.line(band, (int(round(xc)), int(round(yc))), (int(round(ex)), foot), 1, 5)
    out[r0:r1, c0:c1] = band & win
    return out


def drop_rulers(solid: np.ndarray, min_len: int, max_rise: int = 4) -> np.ndarray:
    """Remove dead-straight horizontal strokes longer than `min_len`: border
    and grid remnants, axis lines, title rules. A drawn surface is never that
    flat for that long (a 2 % cross slope alone rises several pixels over
    40 ft). Tick marks hanging off an axis are ignored when judging flatness
    -- only the horizontal structure is measured -- and removed with it."""
    if not solid.any():
        return solid
    horiz = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(horiz, 8)
    ruler = np.zeros(n, bool)
    ruler[1:] = (st[1:, cv2.CC_STAT_WIDTH] >= min_len) & (st[1:, cv2.CC_STAT_HEIGHT] <= max_rise)
    if not ruler.any():
        return solid
    kill = ruler[lab].astype(np.uint8)
    # take the ticks attached to a ruler with it
    kill = cv2.dilate(kill, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(solid, 8)
    hit = np.unique(lab2[(kill > 0) & (solid > 0)])
    gone = np.zeros(n2, bool)
    gone[hit[hit > 0]] = True
    # only drop whole components that are mostly ruler (keep a template that
    # merely touches an axis line, minus the line itself)
    out = solid.copy()
    for k in np.where(gone)[0]:
        comp = lab2 == k
        if (comp & (kill > 0)).sum() >= 0.5 * comp.sum():
            out[comp] = 0
        else:
            out[comp & (kill > 0)] = 0
    return out
