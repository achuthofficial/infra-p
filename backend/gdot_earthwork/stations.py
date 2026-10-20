"""
Repair station labels across a whole plan set.

Text-layer labels are exact. OCR'd labels on scans are usually right but
not always: a thin italic "1" gets dropped ("4+50" for "14+50"), a "+" is
read as "4+" ("914+50" for "91+50"), or nothing is read at all. A road's
cross-sections are drawn at a regular step (50 ft, 100 ft, ...) and in
order, so the sequence itself tells which reading is right.

The labels are decoded jointly (Viterbi): every section offers its OCR
readings plus an "infer it" option; a reading scores by its OCR votes, and
each consecutive pair scores by whether it steps forward by the road's
regular step. Skipping ahead is allowed (sheets left out) at a small cost;
going backwards (a new road) costs more, so a misread -- or a run of them
that shares the same slip -- loses to the reading, one-character
correction or inference that keeps the sequence regular. Inferred and
corrected labels are marked and reported, never silently trusted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .pages import parse_station


@dataclass
class StationRead:
    label: str | None
    station_ft: float | None
    confidence: float          # 1.0 = text layer or two OCR reads agreeing
    source: str | None         # "text" | "ocr" | "inferred" | None
    note: str | None = None
    candidates: tuple[str, ...] = ()   # alternative OCR readings


def format_station(ft: float, decimals: int = 0) -> str:
    whole = int(ft // 100)
    rest = ft - whole * 100
    return f"{whole}+{rest:0{3 + decimals if decimals else 2}.{decimals}f}"


def _typical_step(reads: list[StationRead]) -> float | None:
    """The road's usual station step, from confidently read labels (or all
    labels when few are confident). Uncertain reads often alternate with good
    ones, so steps are measured between successive reads and divided by the
    number of sections apart."""
    good = [(i, r.station_ft) for i, r in enumerate(reads)
            if r.confidence >= 1.0 and r.station_ft is not None]
    if len(good) < 3:
        good = [(i, r.station_ft) for i, r in enumerate(reads) if r.station_ft is not None]
    diffs = [round((b - a) / (j - i), 2) for (i, a), (j, b) in zip(good, good[1:]) if b > a]
    if not diffs:
        return None
    # most frequent step; on a tie the smaller one (a misread makes an
    # absurdly large jump, never a small one)
    return max(Counter(diffs).items(), key=lambda kv: (kv[1], -kv[0]))[0]


INFER = None          # the "infer this label" option
_INFER_COST = -1.5
_EDIT_SCORE = -1.5
_ALT_SCORE = 0.3
_FULLER_BONUS = 0.4
_MAX_INFER_STATES = 40      # distinct "last real station" histories kept through unreadable labels
_REGULAR, _NEAR, _SKIP, _BREAK = 3.0, 1.5, -2.0, -6.0


def _options(r: StationRead) -> list[tuple[float | None, str | None, float]]:
    """(station_ft, label, emission score) choices for one section."""
    if r.source == "text":
        return [(r.station_ft, r.label, 0.0)]
    opts: dict[float, tuple[float, str, float]] = {}
    if r.station_ft is not None:
        opts[r.station_ft] = (r.station_ft, r.label, 1.0 if r.confidence >= 1.0 else 0.5)
    for c in r.candidates:
        ft = parse_station(c)
        if ft is not None and ft not in opts:
            opts[ft] = (ft, c, _ALT_SCORE)
    # Thin CAD strokes lose digits far more often than OCR invents them: of
    # two readings one digit apart ("102+50" / "12+50") the fuller one gets
    # a small edge, so it wins when the sequence fits either.
    readings = {c for c in [r.label, *r.candidates] if c}
    for ft, (_, label, score) in list(opts.items()):
        if any(_drops_one_digit(label, other) for other in readings):
            opts[ft] = (ft, label, score + _FULLER_BONUS)
    # OCR's typical slips on these fonts are single-character ones before
    # the plus sign: a dropped or misread digit ("72+50" -> "12+50",
    # "175+50" -> "75+50", "108+00" -> "198+00") or an extra one ("301+50"
    # -> "3014+50"). Those edits are offered at a low score, so a clean
    # sequence can win the right reading back.
    for c in [r.label, *r.candidates]:
        for fixed in _single_edits(c):
            ft = parse_station(fixed)
            if ft is not None and ft not in opts:
                opts[ft] = (ft, fixed, _EDIT_SCORE)
    return list(opts.values()) + [(INFER, None, _INFER_COST)]


def _drops_one_digit(full: str, short: str) -> bool:
    """Whether `short` is `full` with one digit missing."""
    a, b = full.replace(".", ""), short.replace(".", "")
    return len(a) == len(b) + 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a)) if a[i].isdigit())


def _single_edits(label: str | None) -> list[str]:
    if not label or "+" not in label:
        return []
    head, tail = label.split("+", 1)
    if not head.isdigit():
        return []
    heads = {head[:-1]} if len(head) >= 2 else set()           # extra digit before "+"
    heads |= {d + head for d in "123456789"}                   # dropped leading digit
    if len(head) >= 2:
        heads.add(head[1:])                                    # spurious leading digit
    heads |= {head[:i] + d + head[i + 1:] for i in range(len(head)) for d in "0123456789"
              if d != head[i]}                                  # a misread digit ("198" for "108", "38" for "39")
    heads |= {head[:i] + d + head[i:] for i in range(1, len(head) + 1) for d in "0123456789"}   # dropped digit
    # a station never starts with 0 (except "0+50")
    out = [f"{h}+{tail}" for h in sorted(heads) if h and (h == "0" or not h.startswith("0"))]
    # a misread digit after the plus ("213+30" for "213+50"), to the 25 ft
    # stations sections are cut at -- an odd one (26+40) is read as it is,
    # never made up to fit a step
    whole, dot, dec = tail.partition(".")
    if len(whole) == 2 and whole.isdigit() and not dec.strip("0"):
        out += [f"{head}+{t}{dot}{dec}" for t in ("00", "25", "50", "75")
                if t != whole and sum(a != b for a, b in zip(t, whole)) == 1]
    return out


def _transition(a: float, b: float, expected: float, step: float) -> float:
    d = b - a
    if abs(d - expected) <= 0.5:
        return _REGULAR
    if 0 < d <= expected + step + 0.5:
        return _NEAR                 # an inserted station (26+40) or a missing section
    if d > 0:
        return _SKIP                 # sheets left out further ahead on the same road
    return _BREAK                    # a new road (or a misread) going backwards


def repair(reads: list[StationRead]) -> list[StationRead]:
    """`reads` must be in document order with each road ascending."""
    step = _typical_step(reads)
    if not step or not reads:
        return reads
    opts = [_options(r) for r in reads]

    # Viterbi over the options of each section. An "infer" option carries no
    # value of its own, so the next real reading is scored against the last
    # real one, expecting one step per section in between -- an inferred gap
    # can't hide a backwards jump around it. Because of that, an inferred
    # section's state is really (infer, last real station): one is kept per
    # distinct last station, or a run of unreadable labels would commit to a
    # history before the next real reading could decide between them.
    # state: (score, option index, last real station, sections since, back pointer)
    states = [(o[2], j, o[0], 0, -1) for j, o in enumerate(opts[0])]
    trail: list[list[tuple]] = [states]
    for i in range(1, len(reads)):
        row = []
        infer: dict = {}
        for j, (ft, _, em) in enumerate(opts[i]):
            if ft is None:
                for k, (sc, _, ref, since, _) in enumerate(states):
                    cand = (sc + em, j, ref, since + 1, k)
                    if ref not in infer or cand[0] > infer[ref][0]:
                        infer[ref] = cand
                continue
            best = None
            for k, (sc, _, ref, since, _) in enumerate(states):
                v = sc + (_transition(ref, ft, step * (since + 1), step) if ref is not None else 0.0)
                if best is None or v > best[0]:
                    best = (v, k)
            row.append((best[0] + em, j, ft, 0, best[1]))
        row += sorted(infer.values(), key=lambda t: -t[0])[:_MAX_INFER_STATES]
        states = row
        trail.append(states)
    k = max(range(len(states)), key=lambda q: states[q][0])
    choice = [0] * len(reads)
    for i in range(len(reads) - 1, -1, -1):
        choice[i] = trail[i][k][1]
        k = trail[i][k][4]

    decimals = 2 if any(r.label and "." in r.label for r in reads) else 0
    out: list[StationRead] = []
    for r, o, c in zip(reads, opts, choice):
        ft, label, _ = o[c]
        if ft is None:
            out.append(StationRead(None, None, 0.0, None, None, r.candidates))
        elif label == r.label:
            out.append(r)
        elif label in r.candidates:              # an alternative OCR reading fitted better
            out.append(StationRead(label, ft, 0.9, "ocr", None, r.candidates))
        else:                                    # a one-character correction: flag it for review
            out.append(StationRead(label, ft, 0.5, "inferred",
                                   f"station read as {r.label or 'unreadable'!r}; corrected to {label} "
                                   f"to fit the {step:g} ft station sequence", r.candidates))
    _fill_inferred(out, reads, step, decimals)
    return out


def _fill_inferred(out: list[StationRead], reads: list[StationRead], step: float, decimals: int) -> None:
    """Give every section decoded as "infer" a station from its neighbours."""
    n = len(out)
    i = 0
    while i < n:
        if out[i].station_ft is not None:
            i += 1
            continue
        j = i
        while j < n and out[j].station_ft is None:
            j += 1                                           # out[i:j] need values
        prev = out[i - 1].station_ft if i > 0 else None
        nxt = out[j].station_ft if j < n else None
        for k in range(i, j):
            if prev is not None:
                ft = prev + (k - i + 1) * step
            elif nxt is not None:
                ft = nxt - (j - k) * step
            else:
                continue
            if ft < 0 or (nxt is not None and prev is not None and ft >= nxt):
                continue                                     # neighbours don't leave room: leave unplaced
            label = format_station(ft, decimals)
            was = reads[k].label or "unreadable"
            out[k] = StationRead(label, ft, 0.5, "inferred",
                                 f"station read as {was!r}; inferred {label} from the "
                                 f"{step:g} ft station sequence", reads[k].candidates)
        i = j
