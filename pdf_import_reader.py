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

# 4-digit document code in the top-right corner
DOC_CODE_RE = re.compile(r"\b(\d{4})\b")

# Top-right corner: 4-digit document code
_TOP_FRAC   = 0.12   # top strip height as a fraction of page height
_RIGHT_FRAC = 0.30   # right strip width as a fraction of page width

# Bottom-right corner: WOO-added within-document page number (raster stamp)
# Tighter than the top-right region to exclude the document's own page footer
# ("Pagina X van Y") which sits just outside this zone.
_BOTTOM_STRIP_FRAC       = 0.06   # bottom strip height as a fraction of page height
_BOTTOM_RIGHT_STRIP_FRAC = 0.12   # right strip width as a fraction of page width


# ---------------------------------------------------------------------------
# Document-code extraction
# ---------------------------------------------------------------------------

def _doc_code_from_words(page) -> str | None:
    """
    Find the 4-digit document code in the top-right header of a pdfplumber
    page by filtering word bounding boxes.

    Primary search: right _RIGHT_FRAC of the top _TOP_FRAC strip.
    Fallback: full-width top strip, picking the rightmost 4-digit match —
    catches codes that sit slightly outside the expected margin.
    """
    top_limit = page.height * _TOP_FRAC
    left_limit = page.width * (1 - _RIGHT_FRAC)

    # Primary: tight right-corner region
    for word in page.extract_words():
        if word["top"] <= top_limit and word["x0"] >= left_limit:
            if re.fullmatch(r"\d{4}", word["text"].strip()):
                return word["text"].strip()

    # Fallback: full top strip — return rightmost 4-digit word
    candidates = [
        (word["x0"], word["text"].strip())
        for word in page.extract_words()
        if word["top"] <= top_limit and re.fullmatch(r"\d{4}", word["text"].strip())
    ]
    if candidates:
        return max(candidates, key=lambda c: c[0])[1]

    return None


def _doc_code_from_image(image: Image.Image) -> str | None:
    """
    Crop the top-right corner of a PIL image and OCR it to find the
    4-digit document code.
    """
    w, h = image.size
    crop = image.crop((
        int(w * (1 - _RIGHT_FRAC)), 0,
        w, int(h * _TOP_FRAC),
    ))
    text = pytesseract.image_to_string(
        crop,
        config="--psm 6 -c tessedit_char_whitelist=0123456789",
    )
    # Primary: clean isolated 4-digit word
    match = DOC_CODE_RE.search(text)
    if match:
        return match.group(1)
    # Fallback: first "0NNN" substring — handles noise-merged output like
    # "542240014" where \b boundaries don't apply mid-digit-run.
    m = re.search(r"(0\d{3})", text)
    return m.group(1) if m else None


def _ocr_bottom_right(image: Image.Image) -> int | None:
    """
    OCR a pre-cropped PIL image of the bottom-right corner and return the
    integer found, or None.  Shared by both the searchable and image-PDF paths.
    """
    text = pytesseract.image_to_string(
        image,
        config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    return int(m.group(1)) if m else None


def _within_doc_page(page) -> int | None:
    """
    Find the WOO-added within-document page number in the bottom-right corner.

    The number is stamped into the raster image layer (not the text overlay),
    so text extraction alone cannot see it.  We render the page and OCR the
    tight bottom-right crop (_BOTTOM_STRIP_FRAC × _BOTTOM_RIGHT_STRIP_FRAC).
    """
    try:
        rendered = page.to_image(resolution=150).original   # PIL Image
        w, h = rendered.size
        crop = rendered.crop((
            int(w * (1 - _BOTTOM_RIGHT_STRIP_FRAC)),
            int(h * (1 - _BOTTOM_STRIP_FRAC)),
            w, h,
        ))
        return _ocr_bottom_right(crop)
    except Exception:
        return None


def _within_doc_page_from_image(image: Image.Image) -> int | None:
    """
    Find the WOO-added within-document page number for image-based PDFs
    (pre-rendered full-page PIL image supplied by the caller).
    """
    w, h = image.size
    crop = image.crop((
        int(w * (1 - _BOTTOM_RIGHT_STRIP_FRAC)),
        int(h * (1 - _BOTTOM_STRIP_FRAC)),
        w, h,
    ))
    return _ocr_bottom_right(crop)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

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
    # Note: \s* (not \s+) between month name and year handles OCR-collapsed
    # text like "november2024" (no space between month and year).
)


def _categorize_document(text: str, first_page: str = "") -> str:
    """
    Classify a WOO document into a category based on keyword scoring.

    *first_page* should be the text of the first page of the document.
    If omitted, the first 1000 characters of *text* are used as a proxy.

    Special scoring rules applied on top of pattern matching:
    - E-mail patterns are scored against *first_page* only: email headers
      always appear at the top; this prevents false positives from embedded
      email addresses or quoted messages inside long reports.
    - Nota gets a +3 bonus if the document opens with "nota" or "notitie",
      since those words rarely appear as the literal first word of a report.
    - Timeline gets a +2 bonus if ≥ 5 date instances are found in the text,
      since timelines are dense with dates even without explicit keywords.
    """
    t = text.lower()
    fp = (first_page or text[:1000]).lower()

    scores: dict[str, int] = {}
    for label, patterns in _CATEGORY_RULES:
        if label == "E-mail":
            scores[label] = sum(1 for p in patterns if re.search(p, fp))
        else:
            scores[label] = sum(1 for p in patterns if re.search(p, t))

    # Bonus: document opens with a distinctive type keyword
    if re.search(r"^\s*(nota|notitie)\b", fp):
        scores["Nota"] = scores.get("Nota", 0) + 3
    if re.search(r"^\s*tijdlijn\b", fp):
        scores["Timeline"] = scores.get("Timeline", 0) + 3

    # Bonus: date-dense documents lean toward Timeline
    if len(_DATE_RE.findall(t)) >= 5:
        scores["Timeline"] = scores.get("Timeline", 0) + 2

    best_label, best_score = "Other", 0
    for label, score in scores.items():
        if score > best_score:
            best_label, best_score = label, score
    return best_label


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: Path = PDF_PATH, ocr_supplement: bool = False) -> dict[str, dict]:
    """
    Extract text from *pdf_path* and group pages by their 4-digit document
    code found in the top-right corner of each page.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file to process.
    ocr_supplement : bool, default False
        When True, every page of a searchable PDF is also rendered at 300 DPI
        and OCR'd.  The redaction-code count per page is taken as the max of
        the text-layer count and the OCR count, so image-only stamps that are
        invisible to the text layer are captured without double-counting.
        Adds roughly 2–3 seconds per page.

    Returns a dict keyed by document code:

        {
            "0001": {
                "doc_code":        str,
                "pages":           list[int],        # 1-based PDF page numbers
                "text":            str,               # concatenated raw text
                "annotated_text":  str,               # text with [REDACTED: …] markers
                "redaction_codes": dict[str, int],    # code → total count across all pages
                "method":          "direct" | "ocr",
            },
            ...
        }

        Pages with no detectable code inherit the code of the preceding page
        (forward-fill), since documents are contiguous blocks in the PDF.
        Only the very first page(s) of the file, if they have no code, are
        grouped under "unknown".
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    searchable = _is_searchable(pdf_path)
    method = "direct" if searchable else "ocr"

    images = None
    if not searchable:
        print("[data_import] Image-based PDF — running OCR (DPI=300, may take a moment).")
        images = convert_from_path(pdf_path, dpi=300)
    else:
        print("[data_import] Searchable PDF — using direct text extraction.")

    if ocr_supplement and searchable:
        print("[data_import] OCR supplement enabled — rendering all pages at 300 DPI.")

    # Accumulate raw data per document code
    docs: dict[str, dict] = defaultdict(
        lambda: {"pages": [], "text_parts": [], "code_counts": Counter(), "page_nums_in_doc": []}
    )

    last_code: str | None = None  # for forward-filling pages with no detectable code
    unknown_count = 0             # counter for new-doc boundaries with no detectable code

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):

            # Detect within-document page number from bottom-right corner.
            # For searchable PDFs this renders the page; we reuse that image
            # below so it is done first.
            if searchable:
                bottom_info = _within_doc_page(page)
            else:
                bottom_info = _within_doc_page_from_image(images[i - 1]) if images else None
            is_new_doc_start = bottom_info == 1

            # Detect document code from header region.
            if searchable:
                detected = _doc_code_from_words(page)      # fast text-layer path
                if is_new_doc_start or detected is None:
                    # Always try raster OCR when:
                    # (a) bottom stamp == 1: text layer can misread (e.g. email
                    #     body "0008" overrides true raster stamp "0006"), or
                    # (b) text layer found nothing: raster may still have the code
                    #     (e.g. page 1 of a new doc whose bottom stamp was unreadable).
                    try:
                        rendered = page.to_image(resolution=200).original
                        raster_code = _doc_code_from_image(rendered)
                        if raster_code:
                            detected = raster_code
                    except Exception:
                        pass
            else:
                detected = _doc_code_from_image(images[i - 1]) if images else None


            inherited = False
            if detected:
                doc_code = detected
                last_code = detected
            elif is_new_doc_start and last_code:
                # Bottom-right shows page 1, but no top-right code → new document boundary
                unknown_count += 1
                doc_code = f"unknown_{unknown_count}"
                last_code = doc_code
            elif last_code:
                # No code found — inherit from the previous page (documents are contiguous)
                doc_code = last_code
                inherited = True
            else:
                # Very first page(s) with no code at all
                doc_code = "unknown"

            # Extract full-page text
            was_ocr_fallback = False
            if searchable:
                text = page.extract_text() or ""
                if len(text.strip()) < 20:
                    # Page has no (or near-empty) text layer — OCR it individually.
                    # Happens for rasterized pages embedded in an otherwise
                    # searchable PDF (e.g. scanned inserts).
                    was_ocr_fallback = True
                    try:
                        ocr_render = page.to_image(resolution=300).original
                        text = _ocr_image(ocr_render)
                    except Exception:
                        pass
            else:
                text = _ocr_image(images[i - 1]) if images else ""

            # Count redaction codes from the text layer (or OCR fallback).
            # Normalize each match to strip spurious extra letters (e.g. '5.1.2el' → '5.1.2e').
            page_codes: Counter = Counter(
                _normalize_code(c) for c in REDACTION_CODE_RE.findall(text)
            )

            # OCR supplement: re-render at 300 DPI and OCR to catch image-only
            # redaction stamps not visible in the text layer.  Take max() per
            # code to avoid double-counting codes already in the text layer.
            if ocr_supplement and searchable and not was_ocr_fallback:
                try:
                    ocr_render = page.to_image(resolution=300).original
                    ocr_text = _ocr_image(ocr_render)
                    ocr_codes: Counter = Counter(
                        _normalize_code(c) for c in REDACTION_CODE_RE.findall(ocr_text)
                    )
                    all_keys = set(page_codes) | set(ocr_codes)
                    page_codes = Counter({k: max(page_codes[k], ocr_codes[k]) for k in all_keys})
                except Exception:
                    pass

            docs[doc_code]["pages"].append(i)
            docs[doc_code]["text_parts"].append(text)
            docs[doc_code]["code_counts"].update(page_codes)
            docs[doc_code]["page_nums_in_doc"].append(bottom_info)

            status = f"doc={doc_code}" + (" (inherited)" if inherited else "")
            if bottom_info is not None:
                status += f"  page-in-doc={bottom_info}"
            if page_codes:
                status += f"  redactions={dict(page_codes)}"
            print(f"  Page {i}: {status}")

    # Build final output, sorted by document code
    result = {}
    for code in sorted(docs):
        data = docs[code]
        combined = "\n\n".join(data["text_parts"])
        first_page = data["text_parts"][0] if data["text_parts"] else ""
        result[code] = {
            "doc_code": code,
            "pages": data["pages"],
            "page_nums_in_doc": data["page_nums_in_doc"],
            "text": combined,
            "annotated_text": _annotate_redactions(combined),
            "redaction_codes": dict(data["code_counts"]),  # {code: total count}
            "category": _categorize_document(combined, first_page),
            "method": method,
        }

    n_docs = len([k for k in result if k != "unknown" and not k.startswith("unknown_")])
    n_pages = sum(len(d["pages"]) for d in result.values())
    print(f"[data_import] Done. {n_pages} page(s) → {n_docs} document(s).")
    return result
