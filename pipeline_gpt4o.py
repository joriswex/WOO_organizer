"""
pipeline_gpt4o.py — Full-page GPT-4o VLM pipeline for WOO documents.

Uses OpenAI GPT-4o vision to:
  1. Extract page text with layout and structure preservation
  2. Detect document boundaries (new document starts)
  3. Classify document type (Email, Nota, Brief, etc.)
  4. Identify document codes and within-doc page numbers

Drop-in alternative to pipeline_ocr.py with the same output schema.
Use main_gpt4o.py as the entry point.

Usage:
    from pipeline_gpt4o import load_pdf_vlm
    docs = load_pdf_vlm(pdf_path, api_key="sk-...", max_pages=10)
"""

import base64
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

OPENAI_MODEL  = "gpt-4o"
_RENDER_DPI   = 200
_MAX_IMG_PX   = 1568      # GPT-4o "high" detail works best ≤ 2048px; 1568 is optimal tile size
_CALL_SLEEP   = 0.15      # seconds between API calls (polite rate limiting)
_MAX_RETRIES  = 3

# WOO redaction code pattern — same as pipeline_ocr.py
_REDACTION_RE = re.compile(r"5\.[12]\.[1-9][a-z]{0,2}", re.IGNORECASE)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are analyzing pages from Dutch government documents (WOO — Wet Open Overheid). "
    "Respond ONLY with a valid JSON object matching the schema in the user message. "
    "Do not include markdown code fences or any text outside the JSON object."
)

_PAGE_PROMPT = """\
Analyze this Dutch government document page and return a JSON object with exactly these fields:

{
  "text": "<all text from the page, preserving structure — see rules below>",
  "is_new_document": <true if this is clearly the first page of a new document>,
  "doc_code": "<4-digit stamp code like 0143, or null if not visible>",
  "within_doc_page": <integer — page number within this document, e.g. 1, 2, 3 — or null>,
  "category": "<Email | Chat | Nota | Report | Brief | Timeline | Vergadernotulen | Other>",
  "has_redactions": <true if black boxes or censored areas are visible>,
  "chat_name": "<name of the chat group or contact if this is a Chat page, else null>",
  "chat_messages": <array of message objects if this is a Chat page, else []>
}

TEXT EXTRACTION RULES:
- Extract ALL text in reading order, preserving paragraph structure.
- For email documents, preserve header fields exactly on their own lines:
    Van: ...
    Aan: ...
    CC: ...
    Onderwerp: ...
    Datum: ...
    Verzonden: ...
- For redacted email addresses where only the domain is visible (e.g. a black box before @minbuza.nl):
    write: <[REDACTED]@minbuza.nl>
- For all other redacted sections (black rectangles / censored words):
    write: [REDACTED]
- Include any visible stamps, codes, or page markers.
- Use blank lines to separate paragraphs and sections.

DOCUMENT BOUNDARY RULES:
- is_new_document = true ONLY when: stamp says "Pagina 1 van N", or a completely fresh document starts at the top of the page (new email header, new memo heading, new letter salutation).
- is_new_document = false if within_doc_page is 2 or higher — a continuation page is never a new document.
- doc_code: look for a 4-digit stamp in any corner (format 0001–0999).
  If you see a 7-digit barcode stamp (e.g. 7601430), extract digits 4–6 and prefix with 0 → "0143".
- within_doc_page: if a stamp or footer says "Pagina X van N", return X.

CATEGORY RULES:
- Email: contains Van/Aan/Onderwerp header block
- Chat: screenshot of a messaging app (WhatsApp, Signal, Telegram, SMS, etc.) showing chat bubbles
- Nota: internal memo or briefing note ("Nota", "Memorandum", "Briefing")
- Brief: formal letter with salutation ("Geachte", "Beste", "Dear")
- Report: rapport, onderzoek, analyse
- Timeline: chronological list of events or dates
- Vergadernotulen: meeting minutes or agenda
- Other: anything that does not fit the above

CHAT PAGE RULES (only when category = "Chat"):
- Set chat_name to the group or contact name shown at the top of the screen (e.g. "Bezoek praktische zaken").
- Extract every visible message into chat_messages as an array of objects:
  {
    "sender_position": "left" | "right",  // left = incoming, right = outgoing (device owner)
    "sender_label": "<name or redaction code visible above the bubble, e.g. '5.1.2.e' or '[REDACTED]'>",
    "timestamp": "<time string visible on the message, e.g. '13:27', or null>",
    "content": "<full message text; write [REDACTED] for blacked-out words>"
  }
- Preserve message order top to bottom.
- If a sender name is redacted but the same redaction code appears consistently (e.g. always '5.1.2.e'),
  use that code as the sender_label so we can track the same person across messages.
- For the text field, write a plain concatenation of all message contents (for search/sorting).
"""


# ── Category normalisation ────────────────────────────────────────────────────
# Maps GPT-4o category labels → canonical labels used by visualisation.py
_GPT4O_TO_CATEGORY: dict[str, str] = {
    # GPT-4o raw labels
    "email":           "E-mail",
    "Email":           "E-mail",
    "chat":            "Chat",
    "Chat":            "Chat",
    "nota":            "Nota",
    "report":          "Report",
    "brief":           "Brief",
    "timeline":        "Timeline",
    "vergadernotulen": "Vergadernotulen",
    "other":           "Other",
    # Canonical pass-through (already normalised, e.g. loaded from cache)
    "E-mail":          "E-mail",
    "Nota":            "Nota",
    "Report":          "Report",
    "Brief":           "Brief",
    "Timeline":        "Timeline",
    "Vergadernotulen": "Vergadernotulen",
    "Other":           "Other",
}


def _normalise_category(raw: str) -> str:
    return _GPT4O_TO_CATEGORY.get(raw.strip(), "Other")


# ── Image helpers ─────────────────────────────────────────────────────────────

def _encode_image(img: Image.Image) -> str:
    """Resize to optimal GPT-4o size and base64-encode as JPEG."""
    w, h = img.size
    if max(w, h) > _MAX_IMG_PX:
        scale = _MAX_IMG_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ── GPT-4o call ───────────────────────────────────────────────────────────────

def _call_gpt4o(image_b64: str, client, page_index: int) -> dict:
    """Call GPT-4o vision with retry logic. Returns parsed JSON dict."""
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=2048,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": _PAGE_PROMPT},
                        ],
                    },
                ],
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  [gpt4o] p{page_index+1}: JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            print(f"  [gpt4o] p{page_index+1}: API error (attempt {attempt+1}): {err[:120]}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return {}


# ── Text helpers ──────────────────────────────────────────────────────────────

def _count_redaction_codes(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for m in _REDACTION_RE.finditer(text):
        counts[m.group().lower()] += 1
    return dict(counts)


def _annotate_text(text: str) -> str:
    return _REDACTION_RE.sub(lambda m: f"[REDACTED: {m.group()}]", text)


def _normalise_doc_code(raw) -> str | None:
    """Validate and zero-pad a doc code string. Returns '0143' style or None."""
    if not raw:
        return None
    s = re.sub(r"\D", "", str(raw))   # digits only
    if len(s) == 7:
        # 7-digit barcode: extract digits 4-6 and prefix with 0
        s = "0" + s[3:6]
    elif len(s) >= 4:
        s = s[-4:].zfill(4)
    else:
        return None
    # Must be WOO-style 0NNN (leading zero, not a year)
    if re.fullmatch(r"0\d{3}", s):
        return s
    return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def save_cache(page_data: list[dict], cache_path: Path) -> None:
    """Save per-page text/metadata to JSON. Images are excluded (re-rendered on load)."""
    payload = [
        {
            "page_num":        p["page_num"],
            "text":            p["text"],
            "doc_code":        p["doc_code"],
            "is_new_document": p["is_new_document"],
            "within_doc_page": p["within_doc_page"],
            "category":        p["category"],
            "chat_name":       p.get("chat_name"),
            "chat_messages":   p.get("chat_messages") or [],
        }
        for p in page_data
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"dpi": _RENDER_DPI, "pages": payload}, f, ensure_ascii=False, indent=2)
    print(f"[gpt4o] Cache saved → {cache_path}")


def _load_cache_pages(cache_path: Path) -> tuple[list[dict], int]:
    """Load page metadata from JSON cache. Returns (pages_without_images, dpi)."""
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    dpi   = data.get("dpi", _RENDER_DPI)
    pages = data["pages"]
    # Re-normalise category in case cache was written before the fix
    for p in pages:
        p["category"] = _normalise_category(str(p.get("category") or "Other"))
    return pages, dpi


def docs_from_cache(cache_path: Path, pdf_path: Path) -> dict[str, dict]:
    """
    Rebuild the docs dict from a saved JSON cache without calling the API.
    Re-renders page images from the PDF at the original DPI.

    Args:
        cache_path: Path to the JSON cache file written by load_pdf_vlm().
        pdf_path:   Original PDF (needed to re-render thumbnails).

    Returns:
        Same dict[str, dict] as load_pdf_vlm().
    """
    print(f"[gpt4o] Loading cache from {cache_path} ...")
    pages_meta, dpi = _load_cache_pages(cache_path)
    total = len(pages_meta)

    print(f"[gpt4o] Re-rendering {total} page images from PDF at {dpi} DPI...")
    from pdf2image import convert_from_path
    all_images: list[Image.Image] = convert_from_path(str(pdf_path), dpi=dpi)

    # Attach images to page metadata (cache stores 1-based page_num)
    page_data: list[dict] = []
    for p in pages_meta:
        idx = p["page_num"] - 1
        img = all_images[idx] if idx < len(all_images) else all_images[-1]
        page_data.append({**p, "image": img})

    return _finalise_pipeline(page_data)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_pdf_vlm(
    pdf_path:  Path,
    api_key:   str | None = None,
    max_pages: int | None = None,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """
    GPT-4o full-page VLM pipeline. Drop-in replacement for load_pdf().

    Args:
        pdf_path:   Path to the PDF file.
        api_key:    OpenAI API key (falls back to OPENAI_API_KEY env var).
        max_pages:  Only process the first N pages (test mode).
        cache_path: If given, save extracted page data to this JSON file after the run.

    Returns:
        dict[str, dict] with the same schema as pipeline_ocr.load_pdf().
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No OpenAI API key provided. "
            "Set the OPENAI_API_KEY environment variable or pass api_key=..."
        )

    client = OpenAI(api_key=api_key)

    # ── Convert PDF to images ────────────────────────────────────────────────
    print(f"[gpt4o] Converting PDF to images at {_RENDER_DPI} DPI...")
    from pdf2image import convert_from_path
    images: list[Image.Image] = convert_from_path(str(pdf_path), dpi=_RENDER_DPI)

    if max_pages is not None:
        images = images[:max_pages]
        print(f"[gpt4o] Test mode — processing first {len(images)} pages.")

    total = len(images)
    print(f"[gpt4o] Analysing {total} pages with {OPENAI_MODEL} (detail=high)...")

    # ── Stage 1: per-page GPT-4o analysis ───────────────────────────────────
    page_data: list[dict] = []

    for i, img in enumerate(images):
        image_b64 = _encode_image(img)
        result    = _call_gpt4o(image_b64, client, i)

        text            = str(result.get("text") or "").strip()
        is_new_doc      = bool(result.get("is_new_document", False))
        doc_code        = _normalise_doc_code(result.get("doc_code"))
        within_doc_page = result.get("within_doc_page")
        category        = _normalise_category(str(result.get("category") or "Other"))

        if within_doc_page is not None:
            try:
                within_doc_page = int(within_doc_page)
            except (ValueError, TypeError):
                within_doc_page = None

        chat_name     = result.get("chat_name") or None
        chat_messages = result.get("chat_messages") or []
        if not isinstance(chat_messages, list):
            chat_messages = []

        page_data.append({
            "page_num":        i + 1,
            "image":           img,
            "text":            text,
            "doc_code":        doc_code,
            "is_new_document": is_new_doc,
            "within_doc_page": within_doc_page,
            "category":        category,
            "chat_name":       chat_name,
            "chat_messages":   chat_messages,
        })

        status = (
            f"code={doc_code or '?':>4} | "
            f"wpn={str(within_doc_page or '?'):>3} | "
            f"new={str(is_new_doc):<5} | "
            f"cat={category}"
        )
        print(f"[gpt4o] Page {i+1:>{len(str(total))}}/{total} — {status}")
        time.sleep(_CALL_SLEEP)

    # ── Save cache ───────────────────────────────────────────────────────────
    if cache_path:
        save_cache(page_data, cache_path)

    return _finalise_pipeline(page_data)


def _finalise_pipeline(page_data: list[dict]) -> dict[str, dict]:
    """Stage 2 + 3: assign doc codes and build the final docs dict."""
    has_stamps = any(p["doc_code"] for p in page_data)

    if has_stamps:
        docs_raw = _build_docs_forward_fill(page_data)
        method   = "gpt4o-stamp"
    else:
        print("[gpt4o] No doc-code stamps found — using VLM boundary detection.")
        docs_raw = _build_docs_boundary(page_data)
        method   = "gpt4o-boundary"

    docs: dict[str, dict] = {}
    for code, d in docs_raw.items():
        full_text       = "\n\n".join(d["texts"])
        annotated       = _annotate_text(full_text)
        redaction_codes = _count_redaction_codes(full_text)

        cat_counts = Counter(d["categories"])
        category   = cat_counts.most_common(1)[0][0] if cat_counts else "Other"

        # Chat: use most common chat name across pages
        chat_names    = d.get("chat_names") or []
        chat_name     = Counter(chat_names).most_common(1)[0][0] if chat_names else None
        chat_messages = d.get("chat_messages") or []

        docs[code] = {
            "doc_code":         code,
            "pages":            d["pages"],
            "page_nums_in_doc": d["page_nums_in_doc"],
            "text":             full_text,
            "annotated_text":   annotated,
            "redaction_codes":  redaction_codes,
            "category":         category,
            "method":           method,
            "chat_name":        chat_name,
            "chat_messages":    chat_messages,
        }

    total = len(page_data)
    print(f"[gpt4o] Done — {len(docs)} documents from {total} pages.")
    return docs


# ── Assignment helpers ────────────────────────────────────────────────────────

def _build_docs_forward_fill(page_data: list[dict]) -> dict[str, dict]:
    """Stamp-based assignment with forward-fill. Same logic as pipeline_ocr."""
    docs_raw: dict[str, dict] = {}
    last_code     = None
    unknown_count = 0

    for p in page_data:
        detected = p["doc_code"]
        wpn      = p["within_doc_page"]
        # Never start a new doc if VLM explicitly says this is page 2+
        is_new   = (p["is_new_document"] or wpn == 1) and (wpn is None or wpn == 1)

        if detected:
            current_code = detected
            last_code    = detected
        elif is_new and last_code is not None:
            # Boundary but no new code — create unknown slot
            unknown_count += 1
            current_code  = f"unknown_{unknown_count}"
            last_code     = current_code
        else:
            current_code = last_code or "unknown_1"
            if not last_code:
                unknown_count += 1
                current_code  = f"unknown_{unknown_count}"
                last_code     = current_code

        _append_page(docs_raw, current_code, p)

    return docs_raw


def _build_docs_boundary(page_data: list[dict]) -> dict[str, dict]:
    """Boundary-only assignment (no stamps). Uses VLM is_new_document signal."""
    docs_raw: dict[str, dict] = {}
    doc_index = 0

    for p in page_data:
        is_new = p["is_new_document"] or p["within_doc_page"] == 1
        if is_new or doc_index == 0:
            doc_index += 1
        code = f"auto_{doc_index:03d}"
        _append_page(docs_raw, code, p)

    return docs_raw


def _append_page(docs_raw: dict, code: str, p: dict) -> None:
    if code not in docs_raw:
        docs_raw[code] = {
            "doc_code":         code,
            "pages":            [],
            "page_nums_in_doc": [],
            "texts":            [],
            "categories":       [],
            "chat_names":       [],
            "chat_messages":    [],
        }
    docs_raw[code]["pages"].append(p["image"])
    docs_raw[code]["page_nums_in_doc"].append(p["within_doc_page"])
    docs_raw[code]["texts"].append(p["text"])
    docs_raw[code]["categories"].append(p["category"])
    if p.get("chat_name"):
        docs_raw[code]["chat_names"].append(p["chat_name"])
    docs_raw[code]["chat_messages"].extend(p.get("chat_messages") or [])
