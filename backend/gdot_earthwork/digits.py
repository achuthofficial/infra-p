"""
Read a station label ("12+50", "105+25", "97+50.00") glyph by glyph.

Tesseract reads these short labels poorly: the thin "+" between two digit
groups comes out as a "4" ("106400"), or throws the digit beside it ("196+00"
for "106+00"), and a lone digit is often not read at all. A station label has
a fixed shape, though -- digits, a plus, two digits, maybe a decimal part --
and its characters stand apart from one another. So the label is cut into its
characters, the plus and the decimal point are told by their shape, and each
digit is matched against digits rendered in the fonts plan sets are lettered
in (Helvetica / Arial, Courier, Times, and the single-stroke Hershey fonts
CAD "romans" lettering is drawn from), upright and slanted. Glyphs are
compared by the distance between their centre lines (chamfer), so a heavy pen
and a fine one match alike.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import pymupdf as fitz

SIZE = 32          # glyphs are compared on a SIZE x SIZE canvas
SLANTS = (0.0, 0.22)
HOLE_COST = 0.6    # chamfer px a missing / extra hole is worth


def _skeleton(ink: np.ndarray) -> np.ndarray:
    """Morphological skeleton of a binary glyph (uint8 0/1)."""
    img = ink.copy()
    skel = np.zeros_like(img)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(SIZE):                 # paper beyond the edge: ink touching it still erodes
        if not img.any():
            break
        eroded = cv2.erode(img, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        skel |= img & ~cv2.dilate(eroded, k)
        img = eroded
    return skel


def _normalise(ink: np.ndarray) -> np.ndarray | None:
    """A glyph's ink cropped, scaled to SIZE high (aspect kept), centred."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 3:
        return None
    g = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = g.shape
    f = SIZE / h
    nw = max(1, min(SIZE, int(round(w * f))))
    g = cv2.resize(g * 255, (nw, SIZE), interpolation=cv2.INTER_AREA) > 96
    out = np.zeros((SIZE, SIZE), np.uint8)
    x0 = (SIZE - nw) // 2
    out[:, x0:x0 + nw] = g
    return out


def _holes(ink: np.ndarray) -> int:
    """Enclosed holes of a glyph: one in 0, 6, 9 (and a closed 4), two in 8.
    Specks of paper smaller than a stroke's width don't count."""
    paper = np.pad(1 - ink, 1, constant_values=1).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(paper, 4)
    border = lab[0, 0]
    return sum(1 for k in range(1, n) if k != border and st[k, cv2.CC_STAT_AREA] >= 6)


def _signature(ink: np.ndarray):
    """(skeleton, distance-to-skeleton map, width / height, holes) of a
    normalised glyph, smoothed first so a ragged scanned edge doesn't grow
    spurs on its centre line."""
    ink = cv2.medianBlur(ink.astype(np.uint8) * 255, 3) > 0
    ink = ink.astype(np.uint8)
    sk = _skeleton(cv2.dilate(ink, np.ones((2, 2), np.uint8)))
    if not sk.any():
        sk = ink
    dist = cv2.distanceTransform((1 - sk).astype(np.uint8), cv2.DIST_L2, 3)
    cols = np.flatnonzero(ink.any(axis=0))
    aspect = (cols[-1] - cols[0] + 1) / SIZE if len(cols) else 0.0
    return sk.astype(bool), dist, aspect, _holes(ink)


def _render_fitz(ch: str, font: str) -> np.ndarray:
    doc = fitz.open()
    page = doc.new_page(width=80, height=80)
    page.insert_text((10, 60), ch, fontsize=48, fontname=font)
    pix = page.get_pixmap(dpi=144, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width) < 128


def _render_hershey(ch: str, font: int, thick: int) -> np.ndarray:
    img = np.zeros((120, 120), np.uint8)
    cv2.putText(img, ch, (20, 95), font, 2.6, 1, thick, cv2.LINE_8)
    return img > 0


def _slant(ink: np.ndarray, s: float) -> np.ndarray:
    if not s:
        return ink
    h, w = ink.shape
    M = np.float32([[1, -s, s * h], [0, 1, 0]])          # top leans right, as italic lettering
    return cv2.warpAffine(ink.astype(np.uint8), M, (int(w + s * h) + 2, h), flags=cv2.INTER_NEAREST) > 0


@lru_cache(maxsize=1)
def _templates() -> list[tuple[str, tuple]]:
    out = []
    renders = []
    for font in ("helv", "hebo", "cour", "cobo", "tiro", "tibo"):
        renders.append(lambda ch, f=font: _render_fitz(ch, f))
    for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN):
        renders.append(lambda ch, f=font: _render_hershey(ch, f, 3))
    for render in renders:
        for ch in "0123456789":
            base = render(ch)
            for s in SLANTS:
                g = _normalise(_slant(base, s))
                if g is not None:
                    out.append((ch, _signature(g)))
    return out


def classify(ink: np.ndarray) -> tuple[str, float]:
    """The digit a glyph's ink (binary) is, and the margin to the next-best
    digit (0 = a toss-up, larger = surer)."""
    g = _normalise(ink)
    if g is None:
        return "?", 0.0
    sk, dist, aspect, holes = _signature(g)
    best: dict[str, float] = {}
    for ch, (tsk, tdist, tasp, tholes) in _templates():
        # symmetric chamfer: each centre line's mean distance to the other's;
        # a hole more or less (a 5 against a 6) counts as a near miss
        d = 0.5 * (tdist[sk].mean() + dist[tsk].mean()) + 4.0 * abs(aspect - tasp) + HOLE_COST * abs(holes - tholes)
        if d < best.get(ch, np.inf):
            best[ch] = d
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    (c1, d1), (_, d2) = ranked[0], ranked[1]
    return c1, (d2 - d1) / max(d2, 1e-6)


def _components(ink: np.ndarray) -> list[list[int]]:
    """Characters of a word, left to right: [x, y, w, h], with the pieces of a
    broken glyph merged and glyphs that touch split apart."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    comps = sorted([list(map(int, st[k, :4])) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 4],
                   key=lambda c: c[0])
    if not comps:
        return []
    H = float(np.percentile([c[3] for c in comps], 75))        # a character's height
    merged: list[list[int]] = []
    for c in comps:
        if merged:
            m = merged[-1]
            overlap = min(m[0] + m[2], c[0] + c[2]) - max(m[0], c[0])
            y0, y1 = min(m[1], c[1]), max(m[1] + m[3], c[1] + c[3])
            # pieces of one glyph broken apart, one above the other -- not a
            # stroke above or below the lettering (a dash of the ground line)
            # or a stray arm of a thin "+" (or a nick of a glyph) beside it,
            # small and off the baseline, where a decimal point would sit
            small = min(m, c, key=lambda q: q[3])
            arm = (small[3] < 0.5 * H and overlap >= -2 and small[1] + small[3] < max(m[1] + m[3], c[1] + c[3]) - 0.15 * H
                   and y1 - y0 <= 1.15 * H)
            if (overlap >= 0.5 * min(m[2], c[2]) and y1 - y0 <= 1.15 * H) or arm:
                x0 = min(m[0], c[0])
                m[:] = [x0, y0, max(m[0] + m[2], c[0] + c[2]) - x0, y1 - y0]
                continue
        merged.append(c)
    out = []
    for c in merged:
        x, y, w, h = c
        if h >= 0.75 * H and w > 1.15 * h and not _is_plus(ink[y:y + h, x:x + w]):   # digits run together
            k = max(2, int(round(w / (0.62 * h))))
            prof = ink[y:y + h, x:x + w].sum(axis=0).astype(float)
            cuts = [0]
            for j in range(1, k):
                guess = j * w / k
                lo, hi = int(guess - 0.2 * w / k), int(guess + 0.2 * w / k) + 1
                cuts.append(lo + int(np.argmin(prof[lo:hi])))
            cuts.append(w)
            for a, b in zip(cuts, cuts[1:]):
                rows = np.flatnonzero(ink[y:y + h, x + a:x + b].any(axis=1))
                if b - a > 1 and len(rows):               # each part's own height: a "+" run into a "4"
                    out.append([x + a, y + int(rows[0]), b - a, int(rows[-1] - rows[0] + 1)])
        else:
            out.append(c)
    return out


def _is_plus(g: np.ndarray) -> bool:
    """A cross: its middle row and middle column inked nearly across, its
    corners bare (a 4, an 8 or a 0 has ink there)."""
    h, w = g.shape
    if h < 5 or w < 5:
        return False
    rows = g[int(0.35 * h):int(np.ceil(0.65 * h))].any(axis=0).mean()
    cols = g[:, int(0.35 * w):int(np.ceil(0.65 * w))].any(axis=1).mean()
    ch, cw = max(1, int(0.3 * h)), max(1, int(0.3 * w))
    corners = [g[:ch, :cw], g[:ch, -cw:], g[-ch:, :cw], g[-ch:, -cw:]]
    return rows >= 0.75 and cols >= 0.75 and max(c.mean() for c in corners) < 0.08


def read_station(word: np.ndarray, min_margin: float = 0.04) -> tuple[str, float] | None:
    """A station label read glyph by glyph from a word image (dark ink on
    white): (label, confidence 0..1), or None when the word isn't shaped like
    a station label. The confidence is the least sure digit's margin."""
    ink = (word < 128).astype(np.uint8)
    chars = _components(ink)
    if len(chars) < 4:
        return None
    H = max(c[3] for c in chars)
    base = max(c[1] + c[3] for c in chars)
    kinds = []
    for x, y, w, h in chars:
        cy = y + h / 2
        if h < 0.3 * H and w < 0.45 * H and y + h >= base - 0.2 * H:
            kinds.append(".")                                  # a decimal point sits on the baseline
        elif 0.35 * H <= h < 0.95 * H and 0.55 <= w / h <= 1.8 and abs(cy - (base - H / 2)) < 0.25 * H \
                and _is_plus(ink[y:y + h, x:x + w]):
            kinds.append("+")                                  # a cross, mid-height, no taller than a digit
        elif h >= 0.75 * H:
            kinds.append("d")
        else:
            kinds.append(None)                                 # a speck or a stray stroke
    keep = [(c, k) for c, k in zip(chars, kinds) if k]
    pattern = "".join(k for _, k in keep)
    if pattern.count("+") != 1:
        return None
    head, _, tail = pattern.partition("+")
    if not (1 <= len(head) <= 4 and set(head) == {"d"} and tail in ("dd", "dd.dd", "dd.d")):
        return None
    label, worst = "", 1.0
    for (x, y, w, h), k in keep:
        if k != "d":
            label += k
            continue
        ch, margin = classify(ink[y:y + h, x:x + w])
        label += ch
        worst = min(worst, margin)
    if worst < min_margin:
        return None
    return label, worst
