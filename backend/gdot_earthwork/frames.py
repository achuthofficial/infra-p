"""
Find every cross-section "frame" on a page image and read its station.

A frame is the gridded plot box of one cross-section. It is found purely
from image structure, so the same code serves scanned and vector sheets,
one section per page (19 series strips) or several stacked per sheet
(23 series, highway_planning1):

1. Long horizontal grid lines are pulled out with a morphological open.
   The minor 1-2 ft grid of 23-series sheets is printed light grey and
   falls above the ink threshold, so only major (10 ft) lines remain.
2. Lines are chained into frames: consecutive lines one grid pitch apart
   belong to the same frame; a larger gap starts a new frame. Sheet
   borders and title-block rules don't repeat at the grid pitch, so they
   never form a chain.
3. Inside each frame's band the full-height vertical lines give the plot's
   left/right edges and the horizontal pitch.

A frame only counts as a cross-section when a station label (NNN+NN) is
found inside it -- from the PDF text layer when there is one, otherwise
by OCR of the frame's right-hand side. Plan, profile and notes sheets
fail that test and are skipped; that IS the page classification.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .digits import read_station
from .pages import PageImage, parse_station
from .tracing import strip_grid

SURE_MARGIN = 0.15      # digits.read_station margin at which its reading stands alone
OCR_STATION_RE = re.compile(r"(?<![\d.])(\d{1,4})[ ]?\+[ ]?(\d{2})(?:[ ]?\.[ ]?(\d{1,2}))?(?![\d])")


@dataclass
class Frame:
    page: int
    top: int                 # pixel rows/cols of the outer major grid lines
    bottom: int
    left: int
    right: int
    pitch_x: float           # pixels per major grid square, each axis
    pitch_y: float
    hlines: list[tuple[int, int]] = field(default_factory=list)   # (start, end) rows
    vlines: list[tuple[int, int]] = field(default_factory=list)   # (start, end) cols
    station_label: str | None = None
    station_ft: float | None = None
    label_source: str | None = None   # "text", "ocr" or "inferred"
    label_confidence: float = 0.0
    label_candidates: list[str] = field(default_factory=list)   # all OCR readings

    pad_top: int = 0         # extra rows measured above / below the grid: a
    pad_bottom: int = 0      # drawing may spill past it (a ditch below the axis)
    # real-world size of one major grid square, per axis (stages.calibrate reads
    # them off the axis labels; 10 ft until it does)
    ft_x: float = 10.0
    ft_y: float = 10.0

    @property
    def y0(self) -> int:
        return self.top - self.pad_top

    @property
    def y1(self) -> int:
        return self.bottom + self.pad_bottom

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _runs(idx: np.ndarray, tol: int = 2) -> list[tuple[int, int]]:
    if len(idx) == 0:
        return []
    out, s, p = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - p > tol:
            out.append((s, p))
            s = v
        p = v
    out.append((s, p))
    return out


def _mode_gap(values: list[float], bin_px: float = 2.0) -> float | None:
    if not values:
        return None
    c = Counter(round(v / bin_px) for v in values)
    mode_bin = c.most_common(1)[0][0]
    near = [v for v in values if abs(v / bin_px - mode_bin) <= 1]
    return float(np.median(near))


def horizontal_lines(ink: np.ndarray, min_frac: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Rows carrying a long horizontal stroke: (y0, y1, x_start, x_end)."""
    H, W = ink.shape
    k = max(40, int(W * min_frac))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    rows = np.where(opened.sum(1) >= k)[0]
    out = []
    for y0, y1 in _runs(rows, tol=1):
        band = opened[y0:y1 + 1].max(0)
        xs = np.where(band)[0]
        out.append((y0, y1, int(xs.min()), int(xs.max())))
    return out


def find_frames(img: PageImage, ink_thresh: int = 200, tol: float = 0.12) -> list[Frame]:
    gray = img.gray
    ink = (gray < ink_thresh).astype(np.uint8)
    lines = horizontal_lines(ink)
    if len(lines) < 3:
        return []
    ys = [(a + b) / 2 for a, b, _, _ in lines]
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 8]
    pitch = _mode_gap(gaps)
    if not pitch:
        return []

    # Chain lines spaced one pitch apart into frames.
    chains: list[list[int]] = []
    used = set()
    for i in range(len(lines)):
        if i in used:
            continue
        chain = [i]
        j = i
        while True:
            nxt = None
            for k in range(j + 1, len(lines)):
                d = ys[k] - ys[j]
                if d > pitch * (1 + tol):
                    break
                # grids are printed true to well under 6 %; a looser match
                # would chain across the gap between two stacked sections.
                # A rule reaching well past the grid either side (a title
                # block's border one square below the axis) is not a grid line.
                beyond = max(lines[j][2] - lines[k][2], lines[k][3] - lines[j][3])
                if abs(d - pitch) <= pitch * min(tol, 0.06) and _overlap(lines[j], lines[k]) > 0.8 \
                        and beyond <= 0.5 * pitch:
                    nxt = k
                    break
            if nxt is None:
                break
            chain.append(nxt)
            j = nxt
        if len(chain) >= 3:
            chains.append(chain)
            used.update(chain)

    frames = []
    for chain in chains:
        hl = [lines[i] for i in chain]
        top, bottom = hl[0][0], hl[-1][1]
        x0 = int(np.median([ln[2] for ln in hl]))
        x1 = int(np.median([ln[3] for ln in hl]))
        band = ink[top:bottom + 1, x0:x1 + 1]
        full = np.where(band.sum(0) >= 0.9 * band.shape[0])[0]
        vl = _regular_lattice([(a + x0, b + x0) for a, b in _runs(full, tol=1)], tol)
        if len(vl) < 4:
            continue
        pitch_x = _mode_gap([(b[0] + b[1]) / 2 - (a[0] + a[1]) / 2 for a, b in zip(vl, vl[1:])])
        for sub in _split_at_axes(ink, hl, vl):
            if len(sub) < 3:
                continue
            pitch_y = float(np.median(np.diff([(a + b) / 2 for a, b, _, _ in sub])))
            frames.append(Frame(
                page=img.number, top=int(sub[0][0]), bottom=int(sub[-1][1]),
                left=int(vl[0][0]), right=int(vl[-1][1]),
                pitch_x=float(pitch_x), pitch_y=pitch_y,
                hlines=[(a, b) for a, b, _, _ in sub], vlines=vl,
            ))
    return _add_margins(_dedupe(frames), gray.shape[0], lines)


def _add_margins(frames: list[Frame], height: int, lines) -> list[Frame]:
    """Let a section's drawing spill past its grid: 0.6 squares below (ditches
    drawn into the label strip), 0.3 above -- but never into a neighbouring
    section that shares the boundary line, nor past the next long rule on the
    sheet (a title block, a border)."""
    for f in frames:
        rules_below = [a for a, b, x0, x1 in lines if a > f.bottom + 2]
        rules_above = [b for a, b, x0, x1 in lines if b < f.top - 2]
        below = [g.top for g in frames if g is not f and g.top >= f.bottom - 0.25 * f.pitch_y] + rules_below
        above = [g.bottom for g in frames if g is not f and g.bottom <= f.top + 0.25 * f.pitch_y] + rules_above
        room_b = (min(below) - f.bottom) if below else height - 1 - f.bottom
        room_a = (f.top - max(above)) if above else f.top
        # stop just short of the next rule: what lies beyond it (a title block,
        # the next section) is on the far side of a line
        f.pad_bottom = int(max(0, min(0.6 * f.pitch_y, room_b - 4)))
        f.pad_top = int(max(0, min(0.3 * f.pitch_y, room_a - 4)))
    return frames


def _dedupe(frames: list[Frame]) -> list[Frame]:
    """Two chains can describe the same plot (a stray rule just above the top
    grid line starts a second, slightly shifted chain). Keep the one whose
    vertical pitch best matches its horizontal pitch -- a true grid is square."""
    keep: list[Frame] = []
    for f in sorted(frames, key=lambda f: abs(f.pitch_y - f.pitch_x)):
        clash = False
        for k in keep:
            inter = min(f.bottom, k.bottom) - max(f.top, k.top)
            if inter > 0.5 * min(f.height, k.height) and min(f.right, k.right) > max(f.left, k.left):
                clash = True
                break
        if not clash:
            keep.append(f)
    return sorted(keep, key=lambda f: (f.top, f.left))


def _regular_lattice(vl: list[tuple[int, int]], tol: float) -> list[tuple[int, int]]:
    """Keep the evenly spaced grid columns, dropping stray verticals such as
    a sheet border that happens to sit beside the plot."""
    if len(vl) < 4:
        return vl
    cx = np.array([(a + b) / 2 for a, b in vl])
    pitch = _mode_gap(list(np.diff(cx)))
    if not pitch:
        return []
    best: list[int] = []
    for i in range(len(vl)):
        run = [i]
        for j in range(i + 1, len(vl)):
            k = (cx[j] - cx[run[-1]]) / pitch
            if abs(k - 1) < tol:
                run.append(j)
            elif k > 1 + tol:
                break
        if len(run) > len(best):
            best = run
    return [vl[i] for i in best]


def _tick_count(ink: np.ndarray, row0: int, row1: int, x0: int, x1: int) -> int:
    """Number of short VERTICAL strokes in a strip -- the tick marks of an
    axis. A dashed line lying along the strip has many runs too, but its
    strokes are flat, so only columns inked over 40 % of the strip count."""
    if row1 - row0 < 3 or x1 <= x0:
        return 0
    strip = ink[row0:row1, x0:x1]
    tall = strip.sum(0) >= 0.4 * (row1 - row0)
    return len(_runs(np.where(tall)[0], 1))


def _split_at_axes(ink: np.ndarray, hl, vl) -> list[list]:
    """Stacked sections can share one continuous grid (highway_planning1,
    River Side, Hamilton...). Each section's offset axis carries a dense comb
    of tick marks -- several times more strokes than an ordinary grid line has
    crossings with the vertical grid -- so the chain is cut at every such line.
    A line becomes the shared boundary of the sections above and below it."""
    x0, x1 = vl[0][1] + 2, vl[-1][0] - 2
    gap = (hl[1][0] - hl[0][1]) if len(hl) > 1 else 50
    reach = max(6, int(gap * 0.15))
    need = max(60, 3 * len(vl))
    axes = []
    for i, (a, b, _, _) in enumerate(hl):
        above = _tick_count(ink, max(0, a - reach), a - 1, x0, x1)
        below = _tick_count(ink, b + 2, b + 1 + reach, x0, x1)
        if max(above, below) >= need:
            axes.append(i)
    if not axes or axes == [len(hl) - 1]:
        return [hl]
    # every section ends at its own axis; grid continuing below the last
    # axis (or above the first section) is empty sheet, not a section
    starts = [0] + axes[:-1]
    return [hl[a:b + 1] for a, b in zip(starts, axes) if b - a >= 2]


def _overlap(a, b) -> float:
    lo, hi = max(a[2], b[2]), min(a[3], b[3])
    return max(0, hi - lo) / max(1, min(a[3] - a[2], b[3] - b[2]))


# --------------------------------------------------------------- stations

def _label_regions(frame: Frame) -> list[tuple[float, float, float, float]]:
    """Where a station label is printed, most common layout first:
    inside the frame's right-hand quarter (GDOT standard), or centred just
    below the section's offset axis (some consultants' sheets)."""
    p, w = frame.pitch_y, frame.width
    return [
        (frame.left + 0.70 * w, frame.top - 0.15 * p, frame.right + 0.08 * w, frame.bottom + 0.15 * p),
        (frame.left + 0.25 * w, frame.bottom + 0.2 * p, frame.left + 0.75 * w, frame.bottom + 1.0 * p),
    ]


def label_from_text(frame: Frame, img: PageImage) -> tuple[str, float] | None:
    """Station label from the PDF text layer."""
    for x0, y0, x1, y1 in _label_regions(frame):
        best = None
        for s in img.spans:
            ft = parse_station(s.text)
            if ft is None or not (x0 <= s.x <= x1 and y0 <= s.y <= y1):
                continue
            d = abs(s.y - (y0 + y1) / 2)
            if best is None or d < best[0]:
                best = (d, s.text.strip(), ft)
        if best:
            return best[1], best[2]
    return None


_OFFSET_RE = re.compile(r"^-?\d{1,3}$")


def _has_offset_axis(frame: Frame, img: PageImage) -> bool:
    """A cross-section's bottom axis is labelled with offsets (-140 ... 140).
    Profile sheets have a grid and station numbers too, but their bottom axis
    is labelled with stations, so this tells the two apart."""
    lo, hi = frame.bottom - 0.3 * frame.pitch_y, frame.bottom + 0.6 * frame.pitch_y
    vals = set()
    for s in img.spans:
        if lo <= s.y <= hi and frame.left - frame.pitch_x <= s.x <= frame.right + frame.pitch_x \
                and _OFFSET_RE.match(s.text):
            vals.add(int(s.text))
    return len(vals) >= 8 and min(vals) < 0 < max(vals)


def _clean_for_ocr(crop: np.ndarray, ink_thresh: int) -> np.ndarray:
    ink = (crop < ink_thresh).astype(np.uint8)
    H, W = ink.shape
    for k, shape in ((max(30, W // 3), (1, 0)), (max(30, H // 2), (0, 1))):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1) if shape == (1, 0) else (1, k))
        ink[cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker) > 0] = 0
    # drop tiny specks (minor-grid crossings, tick stubs)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < 6
    ink[small[lab]] = 0
    return (255 - ink * 255).astype(np.uint8)


def _tesseract(img: np.ndarray, psm: str) -> str:
    ok, buf = cv2.imencode(".png", img)
    r = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", psm,
         "-c", "tessedit_char_whitelist=0123456789+."],
        input=buf.tobytes(), capture_output=True,
        # pages are already processed in parallel; tesseract's own OpenMP
        # threads would oversubscribe the CPU and slow every call down
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    return r.stdout.decode("utf-8", "ignore")


def _parse_ocr(txt: str) -> str | None:
    for m in OCR_STATION_RE.finditer(txt):
        label = f"{int(m.group(1))}+{m.group(2)}"
        if m.group(3):
            label += "." + m.group(3).ljust(2, "0")
        return label
    return None


def _word_boxes(ink: np.ndarray, pitch: float) -> list[list[int]]:
    """Group glyph-sized components into words: [x, y, w, h]."""
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    boxes = sorted(
        (tuple(int(v) for v in st[k, :4]) for k in range(1, n)
         if 0.12 * pitch <= st[k, 3] <= 0.45 * pitch and st[k, 2] < 0.4 * pitch),
        key=lambda b: b[0])
    words: list[list[int]] = []
    for x, y, w, h in boxes:
        for wd in words:
            same_line = abs((wd[1] + wd[3] / 2) - (y + h / 2)) < 0.15 * pitch
            # generous gap: a small '+' or '.' is too short to count as a glyph
            # box, so it leaves a hole between "91" and "00.00"
            if same_line and 0 <= x - (wd[0] + wd[2]) < 0.55 * pitch:
                x1 = max(wd[0] + wd[2], x + w)
                y0, y1 = min(wd[1], y), max(wd[1] + wd[3], y + h)
                wd[:] = [wd[0], y0, x1 - wd[0], y1 - y0]
                break
        else:
            words.append([x, y, w, h])
    return words


def _glyph_only(clean: np.ndarray, box, pitch: float, pad: int) -> np.ndarray:
    """The word's crop with everything but glyph-sized strokes whited out.
    CAD masks grid lines around text, so a grid-line stub often abuts a
    label ("|21+75") and is read as a character."""
    x, y, w, h = box
    y0, x0 = max(0, y - pad), max(0, x - pad)
    sub = clean[y0:y + h + pad, x0:x + w + pad]
    ink = (sub < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    drop = np.zeros(n, bool)
    for k in range(1, n):
        cw, ch = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        drop[k] = (ch > 0.5 * pitch and cw <= max(4, 0.06 * pitch)) or \
                  (cw > 0.8 * pitch and ch <= max(4, 0.06 * pitch))       # grid stubs
    return np.where(drop[lab] | (ink == 0), 255, 0).astype(np.uint8)


def _with_grid_strokes(word: np.ndarray, on_grid: np.ndarray, box, pad: int,
                       on_hgrid: np.ndarray | None = None) -> np.ndarray:
    """The word with the grid pixels inside its lettering put back: of the
    vertical grid lines, within its text rows; of the horizontal ones (with
    `on_hgrid`), within the columns of its glyphs only, never across the gaps
    between them. Pieces left standing alone (a grid line passing between two
    glyphs, which would read as a "1") are dropped again: what remains is
    glyph strokes that lie on a grid line -- the side of a "0" on a vertical
    one, the top of a "0" or a "7" on a horizontal one."""
    x, y, w, h = box
    y0, x0 = max(0, y - pad), max(0, x - pad)
    ink = (word < 128).astype(np.uint8)
    sub = on_grid[y0:y0 + word.shape[0], x0:x0 + word.shape[1]].copy()
    sub[:max(0, y - y0)] = 0
    sub[max(0, y - y0) + h:] = 0
    if on_hgrid is not None:
        hs = on_hgrid[y0:y0 + word.shape[0], x0:x0 + word.shape[1]].copy()
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        cols = np.zeros(word.shape[1], bool)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_HEIGHT] >= 0.5 * h:          # a glyph, not a dash of the ground line
                cols[st[k, cv2.CC_STAT_LEFT]:st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH]] = True
        hs[:, ~cols] = 0
        sub |= hs
    both = ink | sub
    n, lab, _, _ = cv2.connectedComponentsWithStats(both, 8)
    alone = np.ones(n, bool)
    alone[np.unique(lab[ink > 0])] = False
    alone[0] = False
    both[alone[lab]] = 0
    return np.where(both > 0, 0, 255).astype(np.uint8)


def _ocr_word(clean: np.ndarray, box, pitch: float, on_grid: np.ndarray | None = None,
              on_hgrid: np.ndarray | None = None) -> list[str]:
    """Readings of one word, by majority over a few renderings.

    CAD lettering is usually italic: shearing it upright is what makes
    tesseract usable on it -- but the same shear spoils an upright font. So
    the word is read sheared and plain, at two glyph heights, plus a
    thickened read, stopping as soon as two readings agree -- after a read
    glyph by glyph (digits.read_station), which tesseract is only asked to
    second when it is unsure of a digit. Returns every
    parseable reading (duplicates kept, so agreement is visible to the
    caller)."""
    x, y, w, h = box
    pad = int(0.1 * pitch)
    word = _glyph_only(clean, box, pitch, pad)
    border = lambda im: cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # read glyph by glyph first (digits.py): sure of every digit, it settles
    # the label as two agreeing reads; otherwise it is one vote among them
    out: list[str] = []
    tries = [read_station(word)]
    if on_grid is not None and on_grid.any():
        tries.append(read_station(_with_grid_strokes(word, on_grid, box, pad)))
    if not any(tries) and on_grid is not None and on_hgrid is not None and on_hgrid.any():
        # only failing both: put back on a horizontal line, a stroke lying
        # along it can as well turn a 5 into a 6 as close the top of a 0
        tries.append(read_station(_with_grid_strokes(word, on_grid, box, pad, on_hgrid)))
    glyphs = max((t for t in tries if t), key=lambda t: t[1], default=None)
    if glyphs:
        out = [glyphs[0]] * (2 if glyphs[1] >= SURE_MARGIN else 1)
        if len(out) >= 2:
            return out

    def render(height: int, shear: bool, thick: bool = False) -> np.ndarray:
        f = height / max(1, h)
        im = cv2.resize(word, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        if shear:
            H, W = im.shape
            im = cv2.warpAffine(im, np.float32([[1, 0.25, 0], [0, 1, 0]]), (int(W + 0.25 * H), H),
                                borderValue=255)
        if thick:
            im = cv2.erode(im, np.ones((3, 3), np.uint8))
        return border(im)

    # tesseract's reading of these fonts depends on glyph size as much as on
    # slant ("1+" becomes "4+" at one size and not another), so vote across both
    for height, shear, thick in ((40, True, False), (40, False, False), (64, True, False),
                                 (64, False, False), (64, True, True)):
        lab = _parse_ocr(_tesseract(render(height, shear, thick), "7"))
        if lab:
            out.append(lab)
            if out.count(lab) >= 2:
                break
    return out


def label_from_ocr(frame: Frame, img: PageImage, ink_thresh: int = 200):
    """OCR the station label (see _label_regions for where it is looked for).
    Returns (label, station_ft, confidence, all readings) or None."""
    if not shutil.which("tesseract"):
        return None
    for x0, y0, x1, y1 in _label_regions(frame):
        got = _ocr_region(frame, img, ink_thresh, int(x0), int(y0), int(x1), int(y1))
        if got is not None:
            return got
    return None


def _ocr_region(frame: Frame, img: PageImage, ink_thresh: int, x0: int, y0: int, x1: int, y1: int):
    H, W = img.gray.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1, frame.right - 3), min(H, y1)   # never the elevation labels past the plot
    crop = img.gray[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None
    # erase the grid at its known position first: CAD masks the grid around
    # text, so the stubs left next to a label are too short to be found as
    # lines. Where a label is lettered over the grid instead, the glyph strokes
    # crossing a grid line are bridged back across it (strip_grid): blanked,
    # a "0" splits in two and the "1" beside the gap reads as "9" or "4".
    h, w = crop.shape
    V = [(a - x0, b - x0) for a, b in frame.vlines if b >= x0 and a - x0 < w]
    Hh = [(a - y0, b - y0) for a, b in frame.hlines if b >= y0 and a - y0 < h]
    raw = (crop < ink_thresh).astype(np.uint8)
    ink = strip_grid(raw, V, Hh, bridge=3)
    clean = _clean_for_ocr(np.where(ink > 0, 0, 255).astype(np.uint8), 128)
    # what lay on the vertical grid lines: a glyph stroke running along one
    # (the side of a "0") is erased with it, and nothing crosses it to bridge
    on_grid = np.zeros_like(raw)
    for a, b in V:
        on_grid[:, max(0, a):b + 1] = raw[:, max(0, a):b + 1]
    on_hgrid = np.zeros_like(raw)
    for a, b in Hh:
        on_hgrid[max(0, a):b + 1, :] = raw[max(0, a):b + 1, :]
    ink = (clean < 128).astype(np.uint8)
    best = None
    # A station ("12+50", "97+50.00") is at least four glyphs wide; elevation
    # labels ("670") and offsets are narrower. Try the widest words first and
    # stop at the first clean read -- OCR is the slow part of a scan.
    words = sorted((b for b in _word_boxes(ink, frame.pitch_y) if b[2] >= 1.8 * b[3]),
                   key=lambda b: -b[2] / b[3])
    for box in words[:4]:
        reads = _ocr_word(clean, box, frame.pitch_y, on_grid, on_hgrid)
        if not reads:
            continue
        # The majority reading wins; failing a majority, the most complete one
        # (thin CAD strokes lose digits more often than they gain them). All
        # readings are kept: the plan-set pass picks whichever fits the station
        # sequence when they disagree (one rendering drops a thin "1", another
        # reads "+" as "4+", and only the neighbouring stations can tell).
        label = max(reads, key=lambda r: (reads.count(r), len(r.replace("+", "").replace(".", ""))))
        conf = 1.0 if reads.count(label) >= 2 else 0.6
        if best is None or conf > best[2]:
            best = (label, parse_station(label), conf, sorted(set(reads)))
        if conf >= 1.0:
            break
    return best


# ------------------------------------------------------------ centreline

_AXIS_NUM_RE = re.compile(r"^-?(\d{1,3})$")


def _vline_index(frame: Frame, x: float) -> int | None:
    """The grid line a centred axis label belongs to, or None if it sits between lines."""
    centres = [(a + b) / 2 for a, b in frame.vlines]
    k = min(range(len(centres)), key=lambda i: abs(centres[i] - x))
    return k if abs(centres[k] - x) < 0.3 * frame.pitch_x else None


def _vote_centre(frame: Frame, reads: list[tuple[int, int, bool]], grid_ft: float) -> float | None:
    """Offset labels are centred on the grid lines and step `grid_ft` per line,
    so each read label names where offset 0 is: (line k, value v) puts it at
    line k - v / grid_ft -- or, when the sign could not be read (a scan's
    minus is too small to OCR reliably), at k + v / grid_ft as well. The line
    named by the most words wins, if at least two and most of them agree."""
    votes: Counter = Counter()
    for k, v, signed in reads:
        if v % grid_ft:
            continue
        steps = int(v // grid_ft)
        votes.update({k - steps} if signed else {k - steps, k + steps})
    if not votes:
        return None
    k0, n = votes.most_common(1)[0]
    # a misread can agree with another by chance: most of what was read must
    # fit the winning line, not just two words
    if n < 2 or n < 0.6 * len(reads) or not 0 <= k0 < len(frame.vlines) or list(votes.values()).count(n) > 1:
        return None
    a, b = frame.vlines[k0]
    return (a + b) / 2


def _axis_row(frame: Frame, ink: np.ndarray) -> tuple[int, int]:
    """The offset axis among the frame's horizontal lines: the one with the
    comb of tick marks (usually the bottom line, but not on every sheet)."""
    x0, x1 = frame.left + 2, frame.right - 2
    reach = max(6, int(0.15 * frame.pitch_y))
    def ticks(line):
        a, b = line
        return max(_tick_count(ink, max(0, a - reach), a - 1, x0, x1), _tick_count(ink, b + 2, b + 1 + reach, x0, x1))
    best = max(frame.hlines, key=ticks) if frame.hlines else (frame.bottom, frame.bottom)
    return best if ticks(best) >= 2 * len(frame.vlines) else (frame.bottom, frame.bottom)


def axis_label_words(frame: Frame, img: PageImage, ink_thresh: int = 200):
    """The offset-axis labels of a scanned section, one word per grid line:
    ([(line index, [x, y, w, h], component ids)], glyph image) -- boxes in the
    returned image's coordinates. Tick marks, minus signs and grid lines are
    left out."""
    H, W = img.gray.shape
    page_ink = (img.gray < ink_thresh).astype(np.uint8)
    a, b = _axis_row(frame, page_ink)
    y0, y1 = max(0, int(a - 0.5 * frame.pitch_y)), min(H, int(b + 0.6 * frame.pitch_y))
    x0, x1 = max(0, int(frame.left - 0.3 * frame.pitch_x)), min(W, int(frame.right + 0.3 * frame.pitch_x))
    crop = img.gray[y0:y1, x0:x1].copy()
    # erase the grid where it is known: each label straddles a vertical grid
    # line, which would otherwise join its glyphs into one tall stroke
    for ha, hb in frame.hlines:
        crop[max(0, ha - y0):max(0, hb - y0 + 1), :] = 255
    for va, vb in frame.vlines:
        crop[:, max(0, va - x0):max(0, vb - x0 + 1)] = 255
    clean = _clean_for_ocr(crop, ink_thresh)
    ink = (clean < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    p = frame.pitch_y
    comps = [k for k in range(1, n) if st[k, 3] <= 0.45 * p and st[k, 2] < 0.4 * p and st[k, 4] >= 6]
    glyph_h = np.median([st[k, 3] for k in comps if st[k, 2] > 0.05 * p] or [0.2 * p])
    # the axis' tick marks: thin, solid, upright strokes clearly shorter than a digit
    tick = lambda k: st[k, 2] <= max(4, 0.035 * p) and st[k, 3] < 0.75 * glyph_h and st[k, 4] >= 0.8 * st[k, 2] * st[k, 3]
    groups: dict[int, list[int]] = {}
    for k in comps:
        if tick(k) or st[k, 3] < 0.5 * glyph_h:          # ticks; minus signs and specks carry no digits
            continue
        line = _vline_index(frame, x0 + st[k, 0] + st[k, 2] / 2)
        if line is not None:
            groups.setdefault(line, []).append(k)
    words = []
    for line, ks in groups.items():
        gx0, gy0 = min(st[k, 0] for k in ks), min(st[k, 1] for k in ks)
        gx1, gy1 = max(st[k, 0] + st[k, 2] for k in ks), max(st[k, 1] + st[k, 3] for k in ks)
        words.append((line, [gx0, gy0, gx1 - gx0, gy1 - gy0], ks))
    glyphs = np.where(np.isin(lab, [k for _, _, ks in words for k in ks]), 0, 255).astype(np.uint8)
    return words, glyphs


def find_centreline(frame: Frame, img: PageImage, ink_thresh: int = 200,
                    grid_ft: float = 10.0) -> tuple[float, str] | None:
    """Image x of the centreline (offset 0), read from the section's offset
    axis labels: from the PDF text layer when there is one ("text"), otherwise
    from the scan ("scan"): first by the labels' widths, failing that by OCR
    of the labels. None when they cannot be read."""
    if img.spans:
        lo, hi = frame.bottom - 0.5 * frame.pitch_y, frame.bottom + 0.6 * frame.pitch_y
        reads = []
        for sp in img.spans:
            t = sp.text.strip()
            if _AXIS_NUM_RE.match(t) and lo <= sp.y <= hi \
                    and frame.left - 0.3 * frame.pitch_x <= sp.x <= frame.right + 0.3 * frame.pitch_x:
                k = _vline_index(frame, sp.x)
                if k is not None:
                    reads.append((k, int(t), True))
        got = _vote_centre(frame, reads, grid_ft)
        if got is not None:
            return got, "text"
    if not shutil.which("tesseract"):
        return None
    words, glyphs = axis_label_words(frame, img, ink_thresh)
    p = frame.pitch_y
    by_width = _centre_from_widths(frame, {line: box[2] for line, box, _ in words}, grid_ft)
    if by_width is not None:
        return by_width, "scan"
    mid = len(frame.vlines) / 2
    words = sorted(words, key=lambda w: abs(w[0] - mid))  # labels near the middle are short and read best
    reads: list[tuple[int, int, bool]] = []
    for line, box, _ in words[:16]:
        got = _ocr_number(glyphs, box, p)
        if got:
            reads.append((line, got[0], False))
            centre = _vote_centre(frame, reads, grid_ft)
            if centre is not None and len(reads) >= 3:
                return centre, "scan"
    centre = _vote_centre(frame, reads, grid_ft)
    return (centre, "scan") if centre is not None else None


def width_fits(frame: Frame, widths: dict[int, float], grid_ft: float) -> list[tuple[float, int]]:
    """How well each candidate centre line explains the axis labels' widths
    for a grid of `grid_ft` per square: [(misfit, line)], best first. "0" is
    one digit wide, 10-90 two, 100 and up three (a minus sign is too small to
    be counted), so each candidate predicts a pattern of widths."""
    lines = sorted(k for k in widths if 0 < k < len(frame.vlines) - 1)   # end labels collide with elevations
    if len(lines) < 8:
        return []
    y = np.array([widths[k] for k in lines], float)
    fits = []
    for k0 in range(len(frame.vlines)):
        digits = np.array([len(f"{abs(k - k0) * grid_ft:g}".replace(".", "")) for k in lines], float)
        if len(set(digits)) < 2:
            continue
        A = np.stack([digits, np.ones_like(digits)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if coef[0] <= 0.3 * np.median(y) / 3:           # a digit must add real width
            continue
        rms = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
        fits.append((rms / coef[0], k0))
    return sorted(fits)


def _centre_from_widths(frame: Frame, widths: dict[int, float], grid_ft: float) -> float | None:
    """Offset 0 from the WIDTHS of the axis labels, no reading needed (see
    width_fits): the candidate line that fits the measured widths best wins,
    if clearly."""
    fits = width_fits(frame, widths, grid_ft)
    if not fits:
        return None
    best, k0 = fits[0]
    if best > 0.25 or (len(fits) > 1 and fits[1][0] < 1.6 * best):
        return None
    a, b = frame.vlines[k0]
    return (a + b) / 2


def _ocr_number(clean: np.ndarray, box, pitch: float) -> list[int]:
    """An axis label's digits (sign ignored), when two renderings agree."""
    x, y, w, h = box
    word = _glyph_only(clean, box, pitch, int(0.1 * pitch))
    reads: list[int] = []
    for height, shear in ((40, True), (40, False), (64, True)):
        f = height / max(1, h)
        im = cv2.resize(word, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        if shear:
            Hh, Ww = im.shape
            im = cv2.warpAffine(im, np.float32([[1, 0.25, 0], [0, 1, 0]]), (int(Ww + 0.25 * Hh), Hh), borderValue=255)
        im = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        m = _AXIS_NUM_RE.match(_tesseract(im, "7").strip().replace("+", "").replace(".", ""))
        if m:
            reads.append(int(m.group(1)))
            if reads.count(reads[-1]) >= 2:
                return [reads[-1]]
    return []


def label_frames(frames: list[Frame], img: PageImage, ink_thresh: int = 200) -> list[Frame]:
    """Attach a station to every frame.

    Vector pages: the text layer is authoritative. A frame counts only with a
    station label AND an offset axis -- that is what separates cross-sections
    from profile sheets, which are gridded and stationed too.

    Scanned pages, and vector pages with no text layer at all (some CAD
    exports draw their lettering as strokes): use a text layer if there is one
    (some scans are OCR'd, others hold a few stray words), else read the label
    by OCR. Frames whose label can't be read are kept: the plan-set pass can
    still infer the station from the neighbouring sections."""
    out = []
    for f in frames:
        got = label_from_text(f, img) if img.spans else None
        if img.kind == "vector" and img.spans:
            if got is None or not _has_offset_axis(f, img):
                continue
            f.station_label, f.station_ft, f.label_source, f.label_confidence = got[0], got[1], "text", 1.0
        elif got is not None:
            f.station_label, f.station_ft, f.label_source, f.label_confidence = got[0], got[1], "text", 1.0
        else:
            ocr = label_from_ocr(f, img, ink_thresh)
            if ocr is not None:
                f.station_label, f.station_ft, f.label_confidence, f.label_candidates = ocr
                f.label_source = "ocr"
        out.append(f)
    return out
