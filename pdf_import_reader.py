import re
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

PDF_PATH = Path(__file__).parent / "test.pdf"

# WOO (Wet open overheid) redaction codes.
# Valid grounds are all under article 5.1.x or 5.2.x, e.g. 5.1.2e, 5.1.1, 5.2.1.
# This intentionally excludes plain decimals (2.5), money amounts (50.000), etc.
REDACTION_CODE_RE = re.compile(r"\b(5\.[12]\.[1-9][a-z]{0,2})\b")

# 4-digit document code
DOC_CODE_RE = re.compile(r"\b(\d{4})\b")

# ---------------------------------------------------------------------------
# Multi-region search constants
# ---------------------------------------------------------------------------

# Fractions used to define search regions as (x0, y0, x1, y1) of page dimensions.
_TOP_FRAC  = 0.12   # top-strip height
_SIDE_FRAC = 0.30   # side-strip width
_BTM_FRAC  = 0.10   # bottom-strip height for doc-code search (wider than page-num strip)

# Text-layer doc-code search: top strip only.
# Body text (email dates, reference numbers) can appear anywhere on the page, so
# searching the bottom via the text layer produces false positives.  Top-strip
# search is safe because document codes always appear in a page header.
_DOC_CODE_TEXT_REGIONS: list[tuple[str, tuple[float, float, float, float]]] = [
    # (name, (x0_frac, y0_frac, x1_frac, y1_frac))
    ("top-right",  (1 - _SIDE_FRAC, 0.00, 1.00,       _TOP_FRAC)),  # primary WOO
    ("top-left",   (0.00,           0.00, _SIDE_FRAC, _TOP_FRAC)),
    ("top-full",   (0.00,           0.00, 1.00,       _TOP_FRAC)),   # widened fallback
]

# Raster-OCR doc-code search: all corners and edges.
# Raster stamps are unambiguous (they contain only the code, nothing else), so
# searching bottom regions is safe.  This covers alternative WOO formats where
# the code appears at the bottom-centre or other corners.
_DOC_CODE_RASTER_REGIONS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("top-right",  (1 - _SIDE_FRAC, 0.00,          1.00,       _TOP_FRAC)),
    ("top-left",   (0.00,           0.00,           _SIDE_FRAC, _TOP_FRAC)),
    ("btm-middle", (0.30,           1 - _BTM_FRAC,  0.70,       1.00)),
    ("btm-right",  (1 - _SIDE_FRAC, 1 - _BTM_FRAC, 1.00,       1.00)),
    ("btm-left",   (0.00,           1 - _BTM_FRAC,  _SIDE_FRAC, 1.00)),
    ("top-full",   (0.00,           0.00,           1.00,       _TOP_FRAC)),
]

# Within-document page-number stamp search: bottom-right only.
# The WOO raster stamp is always placed in the bottom-right corner.
# Searching other bottom regions risks picking up digits from email body text
# (e.g. redaction codes "5.1.2e" → OCR extracts "512" with digits-only filter).
_PAGE_NUM_H_FRAC = 0.06   # tight strip: keeps crop below "Pagina X van Y" text
_PAGE_NUM_W_FRAC = 0.12

_PAGE_NUM_REGIONS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("btm-right",  (1 - _PAGE_NUM_W_FRAC, 1 - _PAGE_NUM_H_FRAC, 1.00, 1.00)),
]


# ---------------------------------------------------------------------------
# Low-level OCR helpers
# ---------------------------------------------------------------------------

def _ocr_bottom_right(image: Image.Image) -> int | None:
    """
    OCR a pre-cropped PIL image and return the integer found, or None.
    Shared by all within-document page-number detection paths.
    """
    text = pytesseract.image_to_string(
        image,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    return int(m.group(1)) if m else None


def _ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image. Prefers Dutch, falls back to English."""
    try:
        return pytesseract.image_to_string(image, lang="nld+eng")
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(image, lang="eng")


def _is_searchable(pdf_path: Path) -> bool:
    """Return True if the PDF has any embedded (selectable) text."""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
    return False


# ---------------------------------------------------------------------------
# Multi-region document-code detection
# ---------------------------------------------------------------------------

def _find_doc_code_words(page) -> str | None:
    """
    Search top-strip text-layer regions for a 4-digit document code.

    Only top regions are searched (see _DOC_CODE_TEXT_REGIONS) to avoid false
    positives from 4-digit numbers in email body text lower on the page.
    Regions are tried in priority order; within each region the rightmost
    4-digit word is returned.
    """
    w, h = page.width, page.height
    words = page.extract_words()
    for _name, (x0f, y0f, x1f, y1f) in _DOC_CODE_TEXT_REGIONS:
        x0, y0, x1, y1 = w * x0f, h * y0f, w * x1f, h * y1f
        candidates = [
            (wd["x0"], wd["text"].strip())
            for wd in words
            if (x0 <= wd["x0"] and wd["x1"] <= x1
                and y0 <= wd["top"] and wd["bottom"] <= y1
                and re.fullmatch(r"\d{4}", wd["text"].strip()))
        ]
        if candidates:
            return max(candidates, key=lambda c: c[0])[1]
    return None


def _find_doc_code_raster(image: Image.Image) -> str | None:
    """
    OCR all corner/edge regions of a rendered page image for a 4-digit document
    code.

    Regions in _DOC_CODE_RASTER_REGIONS are tried in priority order; the first
    unambiguous match wins.  Primary: clean 4-digit match; fallback: leading-zero
    substring (handles noise-merged OCR output like "542240014" → "0014").
    """
    w, h = image.size
    for _name, (x0f, y0f, x1f, y1f) in _DOC_CODE_RASTER_REGIONS:
        crop = image.crop((int(w * x0f), int(h * y0f), int(w * x1f), int(h * y1f)))
        text = pytesseract.image_to_string(
            crop, config="--psm 6 -c tessedit_char_whitelist=0123456789"
        )
        m = DOC_CODE_RE.search(text)
        if m:
            return m.group(1)
        # Fallback: first '0NNN' substring — handles noise-merged strings like "542240014"
        m = re.search(r"(0\d{3})", text)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Multi-region within-document page-number detection
# ---------------------------------------------------------------------------

def _find_within_doc_page_raster(image: Image.Image) -> int | None:
    """
    Search the defined page-number regions of a rendered image for the WOO
    within-document page-number stamp.

    The stamp is a raster image added by WOO (not in the text layer).  Regions
    are searched in priority order; returns the first non-None result.
    """
    w, h = image.size
    for _name, (x0f, y0f, x1f, y1f) in _PAGE_NUM_REGIONS:
        crop = image.crop((int(w * x0f), int(h * y0f), int(w * x1f), int(h * y1f)))
        result = _ocr_bottom_right(crop)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Code normalisation and annotation
# ---------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    """Strip any spurious second letter from an OCR-garbled redaction code.

    WOO article 5 sub-grounds carry at most one letter suffix (e.g. 5.1.2e).
    OCR occasionally appends an extra letter (e.g. '5.1.2el'); we keep only
    the first letter so it counts toward the correct ground.
    """
    return re.sub(r"([a-z])[a-z]+$", r"\1", code)


def _annotate_redactions(text: str) -> str:
    """Replace bare WOO redaction codes with [REDACTED: …] markers."""
    return REDACTION_CODE_RE.sub(
        lambda m: f"[REDACTED: {_normalize_code(m.group(1))}]", text
    )


# ---------------------------------------------------------------------------
# Document categorisation
# ---------------------------------------------------------------------------

# Each entry is (category_label, [regex_patterns]).
# Patterns are matched case-insensitively against the full document text.
# The category with the most matches wins; ties go to the first entry below.
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    # NOTE: E-mail patterns are scored against the FIRST PAGE only (see _categorize_document).
    ("E-mail", [
        # Dutch headers
        r"\bvan:\s",
        r"\baan:\s",
        r"\bonderwerp:\s",
        r"\bverzonden:\s",          # "Verzonden:" = Sent (Dutch Outlook)
        # English headers
        r"\bfrom:\s",
        r"\bto:\s",
        r"\bsubject:\s",
        r"\bsent:\s",               # "Sent:" — Outlook timestamp line
        r"\bcc:\s",
        r"\bbcc:\s",
        # Forwarding / threading markers (both languages)
        r"\boriginal message\b",    # "-----Original Message-----"
        r"\bforwarded message\b",   # "---- Forwarded Message ----"
        r"\bdoorgestuurd bericht\b",
        r"@[\w.-]+\.\w{2,}",       # any email address
    ]),
    ("Nota", [
        r"\bnota\s+aan\b",           # "Nota aan de Minister" — highly distinctive opener
        r"\bnotitie\b",
        r"\bnota\b",
        r"\bter informatie\b",
        r"\bter besluitvorming\b",
        r"\bter advisering\b",
        r"\bdienstonderdeel\b",
        r"\bministerraad\b",
    ]),
    ("Report", [
        r"\brapport\b",
        r"\brapportage\b",
        r"\bonderzoeksrapport\b",
        r"\bsamenvatting\b",
        r"\bconclusie[s]?\b",
        r"\baanbeveling(en)?\b",
        r"\bbevinding(en)?\b",
    ]),
    ("Timeline", [
        r"\btijdlijn\b",
        r"\bchronologie\b",
        r"\bchronologisch\b",
        r"\boverzicht\b.*\bdatum\b",
        r"\bperiode\b",
        r"\b\d{4}\s*[-–]\s*\d{4}\b",  # year ranges like "2020–2021"
    ]),
    ("Vergadernotulen", [
        r"\bnotulen\b",
        r"\bvergadering\b",
        r"\bagendapunt\b",
        r"\bbijeenkomst\b",
        r"\boverleg\b",
        r"\bactions?\b",             # action items in English-language minutes
    ]),
    ("Brief", [
        r"\bgeachte\b",
        r"\bhoogachtend\b",
        r"\bmet vriendelijke groet\b",
        r"\bkenmerk\b",
        r"\bbetreft:\s",
        r"\buw brief\b",
        r"\buw kenmerk\b",
    ]),
]

# Matches common Dutch and ISO date formats for timeline density detection.
_DATE_RE = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"           # dd-mm-yy(yy)
    r"|\b\d{4}[-/]\d{2}[-/]\d{2}\b"                 # yyyy-mm-dd
    r"|\b\d{1,2}\s+(?:jan|feb|mrt|maa|apr|mei|jun|jul|aug|sep|okt|nov|dec)\w*\s*\d{4}\b",
    re.IGNORECASE,
)


def _categorize_document(text: str, first_page: str = "") -> str:
    """
    Classify a WOO document into a category based on keyword scoring.

    *first_page* should be the text of the first page of the document.
    If omitted, the first 1000 characters of *text* are used as a proxy.
    """
    t = text.lower()
    fp = (first_page or text[:1000]).lower()

    scores: dict[str, int] = {}
    for label, patterns in _CATEGORY_RULES:
        if label == "E-mail":
            scores[label] = sum(1 for p in patterns if re.search(p, fp))
        else:
            scores[label] = sum(1 for p in patterns if re.search(p, t))

    if re.search(r"^\s*(nota|notitie)\b", fp):
        scores["Nota"] = scores.get("Nota", 0) + 3
    if re.search(r"^\s*tijdlijn\b", fp):
        scores["Timeline"] = scores.get("Timeline", 0) + 3

    if len(_DATE_RE.findall(t)) >= 5:
        scores["Timeline"] = scores.get("Timeline", 0) + 2

    best_label, best_score = "Other", 0
    for label, score in scores.items():
        if score > best_score:
            best_label, best_score = label, score
    return best_label


# ---------------------------------------------------------------------------
# Auto-split helpers (used when no document stamps are detected)
# ---------------------------------------------------------------------------

def _is_new_doc_boundary(p: dict) -> bool:
    """
    Return True when high-confidence signals indicate a new document starts
    on this page.  Used in both forward-fill and auto-split modes.

    Signals (in priority order):
      1. WOO within-document page stamp == 1
      2. "Pagina 1 van N" footer in the text layer
    """
    if p["within_doc_page"] == 1:
        return True
    if re.search(r"\bpagina\s+1\s+van\b", p["text"], re.IGNORECASE):
        return True
    return False


def _is_fresh_email_start(text: str) -> bool:
    """
    Return True if the first 20 lines of *text* contain at least three distinct
    email header fields, indicating this page starts a fresh email thread.
    """
    count = sum(
        1 for line in text.splitlines()[:20]
        if re.match(
            r"^[ \t]*(?:van|from|aan|to|onderwerp|subject|datum|date|sent|verzonden)\s*:",
            line, re.IGNORECASE,
        )
    )
    return count >= 3


def _auto_split_boundaries(page_data: list[dict]) -> list[bool]:
    """
    Determine document boundaries when no raster stamps were found anywhere.

    Returns a list of booleans (one per page); True means a new document starts
    at that page.

    Priority:
      1. within_doc_page == 1  OR  'Pagina 1 van' in text  (high confidence)
      2. Fresh email header block after a non-email page    (medium confidence)
    """
    n = len(page_data)
    is_new = [False] * n
    is_new[0] = True   # first page always starts a document

    prev_is_email = _is_fresh_email_start(page_data[0]["text"]) if n > 0 else False

    for i in range(1, n):
        p = page_data[i]
        if _is_new_doc_boundary(p):
            is_new[i] = True
        else:
            curr_is_email = _is_fresh_email_start(p["text"])
            if curr_is_email and not prev_is_email:
                is_new[i] = True
            prev_is_email = curr_is_email

    return is_new


# ---------------------------------------------------------------------------
# Document-code assignment helpers
# ---------------------------------------------------------------------------

def _log_page(num: int, code: str, wpn: int | None,
              codes: Counter, inherited: bool) -> None:
    status = f"doc={code}" + (" (inherited)" if inherited else "")
    if wpn is not None:
        status += f"  page-in-doc={wpn}"
    if codes:
        status += f"  redactions={dict(codes)}"
    print(f"  Page {num}: {status}")


def _build_docs_forward_fill(page_data: list[dict]) -> dict:
    """
    Assign document codes using detected stamps with forward-fill fallback.

    Pages whose code is not detectable inherit the previous page's code.
    A page where _is_new_doc_boundary() fires but no code is found gets an
    auto-generated 'unknown_N' code.
    """
    docs: dict = defaultdict(
        lambda: {"pages": [], "text_parts": [], "code_counts": Counter(),
                 "page_nums_in_doc": []}
    )
    last_code: str | None = None
    unknown_count = 0

    for p in page_data:
        i   = p["page_num"]
        det = p["detected_code"]
        wpn = p["within_doc_page"]

        inherited = False
        if det:
            doc_code  = det
            last_code = det
        elif _is_new_doc_boundary(p) and last_code:
            # New document starts here, but no code was readable
            unknown_count += 1
            doc_code  = f"unknown_{unknown_count}"
            last_code = doc_code
        elif last_code:
            doc_code  = last_code
            inherited = True
        else:
            doc_code = "unknown"

        docs[doc_code]["pages"].append(i)
        docs[doc_code]["text_parts"].append(p["text"])
        docs[doc_code]["code_counts"].update(p["code_counts"])
        docs[doc_code]["page_nums_in_doc"].append(wpn)
        _log_page(i, doc_code, wpn, p["code_counts"], inherited)

    return docs


def _build_docs_auto_split(page_data: list[dict]) -> dict:
    """
    Assign auto-generated document codes (auto_001, auto_002, …) based on
    automatically detected document boundaries.
    """
    boundaries = _auto_split_boundaries(page_data)
    docs: dict = defaultdict(
        lambda: {"pages": [], "text_parts": [], "code_counts": Counter(),
                 "page_nums_in_doc": []}
    )
    doc_counter  = 0
    current_code = "auto_001"

    for p, is_new in zip(page_data, boundaries):
        if is_new:
            doc_counter += 1
            current_code = f"auto_{doc_counter:03d}"

        docs[current_code]["pages"].append(p["page_num"])
        docs[current_code]["text_parts"].append(p["text"])
        docs[current_code]["code_counts"].update(p["code_counts"])
        docs[current_code]["page_nums_in_doc"].append(p["within_doc_page"])
        _log_page(p["page_num"], current_code, p["within_doc_page"],
                  p["code_counts"], False)

    return docs


def _finalize_docs(docs_raw: dict, method: str) -> dict[str, dict]:
    """Convert raw per-document accumulator into the final output dict."""
    result = {}
    for code in sorted(docs_raw):
        data     = docs_raw[code]
        combined = "\n\n".join(data["text_parts"])
        first_pg = data["text_parts"][0] if data["text_parts"] else ""
        result[code] = {
            "doc_code":        code,
            "pages":           data["pages"],
            "page_nums_in_doc": data["page_nums_in_doc"],
            "text":            combined,
            "annotated_text":  _annotate_redactions(combined),
            "redaction_codes": dict(data["code_counts"]),
            "category":        _categorize_document(combined, first_pg),
            "method":          method,
        }
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: Path = PDF_PATH, ocr_supplement: bool = False) -> dict[str, dict]:
    """
    Extract text from *pdf_path* and group pages by their 4-digit document code.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file to process.
    ocr_supplement : bool, default False
        When True, every page of a searchable PDF is also rendered at 300 DPI
        and OCR'd.  The redaction-code count per page is taken as the max of
        the text-layer count and the OCR count, capturing image-only stamps
        that are invisible to the text layer.  Adds ~3 s per page.

    Returns
    -------
    dict[str, dict]
        Keyed by document code (4-digit stamp, 'unknown_N', or 'auto_NNN'):

        {
            "0001": {
                "doc_code":         str,
                "pages":            list[int],        # 1-based PDF page numbers
                "page_nums_in_doc": list[int | None], # within-doc page stamps
                "text":             str,               # concatenated raw text
                "annotated_text":   str,               # text with [REDACTED: …] markers
                "redaction_codes":  dict[str, int],    # code → total count
                "category":         str,               # one of 7 categories
                "method":           "direct" | "ocr",
            },
            ...
        }

    Document detection strategy (trust hierarchy)
    ----------------------------------------------
    1. Stamped indicators found  →  forward-fill with 'unknown_N' for readable
       boundaries that lack a code.
    2. No stamps found anywhere  →  auto-split using within-doc page stamps,
       'Pagina 1 van' text, and fresh email-header blocks.

    Region search order
    -------------------
    Doc-code detection searches text-layer regions (_DOC_CODE_TEXT_REGIONS)
    first, then raster regions (_DOC_CODE_RASTER_REGIONS).  Page-number
    detection uses _PAGE_NUM_REGIONS.  All lists are tried in priority order,
    so the pipeline handles PDFs where the stamp is not in the primary WOO
    location.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    searchable = _is_searchable(pdf_path)
    method     = "direct" if searchable else "ocr"

    images = None
    if not searchable:
        print("[data_import] Image-based PDF — running OCR (DPI=300, may take a moment).")
        images = convert_from_path(pdf_path, dpi=300)
    else:
        print("[data_import] Searchable PDF — using direct text extraction.")

    if ocr_supplement and searchable:
        print("[data_import] OCR supplement enabled — rendering all pages at 300 DPI.")

    # ── Stage 1: per-page data collection ────────────────────────────────────
    page_data: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):

            if searchable:
                # One 200 DPI render serves both page-number and doc-code detection,
                # avoiding the previous double-render (150 DPI + 200 DPI).
                rendered = page.to_image(resolution=200).original

                within_doc_page = _find_within_doc_page_raster(rendered)

                detected_code = _find_doc_code_words(page)
                # At a stamp-boundary (page 1 of doc) the text layer can misread
                # the code from email body text — raster OCR corrects this.
                # Also try raster when the text layer found nothing at all.
                if within_doc_page == 1 or detected_code is None:
                    raster = _find_doc_code_raster(rendered)
                    if raster:
                        detected_code = raster

                text    = page.extract_text() or ""
                was_ocr = False
                if len(text.strip()) < 20:
                    # Rasterized page embedded in an otherwise searchable PDF
                    was_ocr = True
                    try:
                        text = _ocr_image(page.to_image(resolution=300).original)
                    except Exception:
                        pass

            else:
                # Fully image-based PDF — everything comes from pre-rendered images
                rendered        = images[i - 1] if images else None
                within_doc_page = _find_within_doc_page_raster(rendered) if rendered else None
                detected_code   = _find_doc_code_raster(rendered) if rendered else None
                text            = _ocr_image(rendered) if rendered else ""
                was_ocr         = True

            # Redaction-code counting (with optional OCR supplement)
            page_codes: Counter = Counter(
                _normalize_code(c) for c in REDACTION_CODE_RE.findall(text)
            )
            if ocr_supplement and searchable and not was_ocr:
                try:
                    hi_res    = page.to_image(resolution=300).original
                    ocr_text  = _ocr_image(hi_res)
                    ocr_codes = Counter(
                        _normalize_code(c) for c in REDACTION_CODE_RE.findall(ocr_text)
                    )
                    all_keys   = set(page_codes) | set(ocr_codes)
                    page_codes = Counter(
                        {k: max(page_codes[k], ocr_codes[k]) for k in all_keys}
                    )
                except Exception:
                    pass

            page_data.append({
                "page_num":        i,
                "detected_code":   detected_code,
                "within_doc_page": within_doc_page,
                "text":            text,
                "code_counts":     page_codes,
            })

    # ── Stage 2: assign document codes ───────────────────────────────────────
    has_stamps = any(p["detected_code"] for p in page_data)
    if has_stamps:
        docs_raw = _build_docs_forward_fill(page_data)
    else:
        print("[data_import] No stamps found — using automatic document splitting.")
        docs_raw = _build_docs_auto_split(page_data)

    result  = _finalize_docs(docs_raw, method)
    n_pages = sum(len(d["pages"]) for d in result.values())
    print(f"[data_import] Done. {n_pages} page(s) → {len(result)} document(s).")
    return result
