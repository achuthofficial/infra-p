"""
Turn one PDF page into the two things the rest of the pipeline needs:

* a grayscale raster of the drawing, and
* the page's text spans and vector strokes, when the PDF has them.

Plan sets come in two physical forms and both must work:

* Scanned / raster sheets (the "19 series" strips and "23 series" full
  sheets): the page is one embedded image. That image is used at its
  NATIVE resolution -- no resampling -- because every measurement
  downstream is made in pixels. Estimators often add a transparent
  markup layer (red/green takeoff shading) on top of the scan; that layer
  is deliberately ignored, it is not part of the drawing.
* Vector CAD exports (e.g. highway_planning1.pdf): there is no scan, so
  the page is rasterized at `vector_dpi` for frame detection and
  overlays, and the real vector strokes are kept for exact surface
  extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pymupdf as fitz

STATION_RE = re.compile(r"^(\d{1,4})\+(\d{2}(?:\.\d+)?)$")
MAX_RENDER_PX = 7200     # longest side of a rendered vector page


@dataclass
class TextSpan:
    x: float        # page-image pixel coords of the span's centre
    y: float
    text: str
    size: float     # font size, in page-image pixels
    w: float = 0.0  # the word's box on the image, when the text layer gives it
    h: float = 0.0


@dataclass
class PageImage:
    number: int                    # 1-based page number in the PDF
    gray: np.ndarray               # uint8 grayscale, white paper / dark ink
    kind: str                      # "raster" or "vector"
    scale: float                   # page-image pixels per PDF point
    spans: list[TextSpan] = field(default_factory=list)
    page: fitz.Page | None = None  # kept for vector stroke access
    rot: int = 0                   # quarter turns applied (np.rot90 k): 0, 1 or -1
    base_shape: tuple[int, int] = (0, 0)   # (H, W) before rotation
    placement: dict | None = None  # how image pixels sit on the PDF page (see page_point)

    def to_view(self, x, y):
        """Map unrotated page-pixel coordinates into this (possibly rotated) image."""
        H, W = self.base_shape if self.rot else self.gray.shape
        if self.rot == 1:                 # counter-clockwise
            return y, W - 1 - x
        if self.rot == -1:                # clockwise
            return H - 1 - y, x
        return x, y

    def page_point(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of the whole loading chain: a pixel of this (possibly
        rotated) image -> PDF page coordinates, unrotated, as PyMuPDF's
        drawing calls expect them."""
        H, W = self.base_shape if self.rot else self.gray.shape
        if self.rot == 1:
            x, y = W - 1 - y, x
        elif self.rot == -1:
            x, y = y, H - 1 - x
        return page_point(self.placement, x, y)

    def rotated(self, k: int) -> PageImage:
        """This page turned a quarter turn (k=1 counter-clockwise, k=-1
        clockwise). Some consultants plot cross-sections sideways on the sheet;
        turned upright they read like any other sheet."""
        base = PageImage(self.number, self.gray, self.kind, self.scale, self.spans, self.page)
        out = PageImage(self.number, np.ascontiguousarray(np.rot90(self.gray, k)), self.kind,
                        self.scale, [], self.page, k, self.gray.shape, self.placement)
        for sp in base.spans:
            x, y = out.to_view(sp.x, sp.y)
            out.spans.append(TextSpan(x, y, sp.text, sp.size, sp.h, sp.w) if k % 2 else TextSpan(x, y, sp.text, sp.size, sp.w, sp.h))
        return out


def parse_station(text: str) -> float | None:
    """'12+50' -> 1250.0, '97+50.00' -> 9750.0, anything else -> None."""
    m = STATION_RE.match(text.strip())
    if not m:
        return None
    return int(m.group(1)) * 100 + float(m.group(2))


def page_point(placement: dict, x: float, y: float) -> tuple[float, float]:
    """Unrotated page image pixel -> unrotated PDF page coordinates.

    vector: the page was rendered as displayed at `zoom`, so pixels are
            displayed points; the page's derotation matrix undoes /Rotate.
    raster: the scan's own placement matrix maps its unit square onto the
            page; `flipped` undoes the quarter turn applied when the image
            was stored sideways."""
    if placement["kind"] == "vector":
        z = placement["zoom"]
        p = fitz.Point(x / z, y / z) * fitz.Matrix(*placement["derot"])
        return p.x, p.y
    w0, h0 = placement["w0"], placement["h0"]
    if placement["flipped"]:              # pixel (x, y) of np.rot90(img, 3) is img[h0 - 1 - x, y]
        x, y = y, h0 - 1 - x
    p = fitz.Point((x + 0.5) / w0, (y + 0.5) / h0) * fitz.Matrix(*placement["m"])
    return p.x, p.y


def _base_image(doc: fitz.Document, page: fitz.Page) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """The page's scan, if the page is essentially one full-page image.

    Picks the largest embedded image whose placement covers most of the
    page. Images carrying a soft mask are transparent overlays (markup),
    not the scan, so they are skipped."""
    best, best_area = None, 0
    page_area = page.rect.width * page.rect.height
    for info in page.get_images(full=True):
        xref, smask = info[0], info[1]
        if smask:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        cover = max(r.width * r.height for r in rects) / page_area
        if cover < 0.5:
            continue
        w, h = info[2], info[3]
        if w * h > best_area:
            best, best_area = xref, w * h
    if best is None:
        return None, None
    pix = fitz.Pixmap(doc, best)
    if pix.alpha or pix.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3:
        # luminance in row blocks: the same arithmetic, without whole-page
        # float64 temporaries (a 6000 x 4000 scan would briefly take ~600 MB)
        rgb, arr = arr, np.empty(arr.shape[:2], np.uint8)
        for r in range(0, rgb.shape[0], 256):
            b = rgb[r:r + 256]
            arr[r:r + 256] = (b[..., 0] * 0.299 + b[..., 1] * 0.587 + b[..., 2] * 0.114).astype(np.uint8)
        del rgb
    else:
        arr = arr[..., 0].copy()
    # Respect the placement orientation: a portrait image placed on a
    # landscape page (or vice versa) has been rotated by the PDF.
    (rect, matrix), = page.get_image_rects(best, transform=True)[:1]
    h0, w0 = arr.shape
    flipped = (rect.width > rect.height) != (w0 > h0)
    if flipped:
        arr = np.ascontiguousarray(np.rot90(arr, 3))
    return arr, {"kind": "raster", "m": tuple(matrix), "w0": w0, "h0": h0, "flipped": flipped}


def _text_spans(page: fitz.Page, scale: float) -> list[TextSpan]:
    """Words of the text layer, in page-image pixels. Words, not spans: CAD
    exports often place "105", "+" and "00" as separate spans, and a station
    label only exists once they are joined."""
    out = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        word = word.strip()
        if word:
            out.append(TextSpan((x0 + x1) / 2 * scale, (y0 + y1) / 2 * scale, word,
                                min(x1 - x0, y1 - y0) * scale, (x1 - x0) * scale, (y1 - y0) * scale))
    return out


def could_hold_sections(doc: fitz.Document, index: int) -> bool:
    """Cheap pre-check before rendering a page. A vector page (a real text
    layer, no full-page scan) can only hold cross-sections if its text has a
    station label somewhere; a scanned page always has to be looked at."""
    page = doc[index]
    words = page.get_text("words")
    if len(words) < 20 or _base_image(doc, page)[0] is not None:
        return True
    return any(STATION_RE.match(w[4].strip()) for w in words)


def sideways_labels(page: fitz.Page) -> bool:
    """True when the page's station labels are printed sideways (sections
    plotted rotated on the sheet) -- a word box taller than it is wide."""
    return any(STATION_RE.match(w.strip()) and (y1 - y0) > (x1 - x0)
               for x0, y0, x1, y1, w, *_ in page.get_text("words"))


def load_page(doc: fitz.Document, index: int, vector_dpi: int = 300) -> PageImage:
    page = doc[index]
    words = page.get_text("words")
    scan, placement = _base_image(doc, page) if len(words) < 20 else (None, None)
    if scan is not None:
        scale = scan.shape[1] / page.rect.width
        return PageImage(index + 1, scan, "raster", scale,
                         _text_spans(page, scale) if words else [], page, placement=placement)
    # cap the render size: a 36x24 in sheet at 300 dpi would be ~78 megapixels
    zoom = min(vector_dpi / 72.0, MAX_RENDER_PX / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()
    return PageImage(index + 1, gray, "vector", zoom, _text_spans(page, zoom), page,
                     placement={"kind": "vector", "zoom": zoom, "derot": tuple(page.derotation_matrix)})
