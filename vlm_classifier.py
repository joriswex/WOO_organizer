"""vlm_classifier.py — VLM-powered page classification using Qwen2.5-VL via Ollama.

Public API:
    is_ollama_available(model)              → bool
    classify_pages(images, page_data, model) → list[dict]

Uses Ollama's REST API at http://localhost:11434 — no Ollama Python library needed.
Memory note: images are resized to max 1200px before encoding to reduce payload size.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time

import requests
from PIL import Image

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL   = "qwen2.5vl:7b"

_MAX_SIDE    = 1200    # px — longest side after resize
_TEMP        = 0.1
_NUM_CTX     = 4096    # hard limit: 8 GB M3
_MAX_RETRIES = 2
_RETRY_DELAY = 2.0     # seconds between retries
_PAGE_SLEEP  = 0.3     # seconds between pages


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaNotAvailableError(RuntimeError):
    """Raised when Ollama is not running or the required model is not pulled."""


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Return True if Ollama is running and *model* is available."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if resp.status_code != 200:
            return False
        data         = resp.json()
        model_prefix = model.split(":")[0]
        return any(
            m.get("name", "").startswith(model_prefix)
            for m in data.get("models", [])
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image(image: Image.Image) -> str:
    """Resize to max _MAX_SIDE px, encode as base64 JPEG string."""
    img = image.copy()
    img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(page_num: int, total_pages: int, prev_summary: str) -> str:
    """Build the classification prompt for one page."""
    return (
        f"You are analyzing page {page_num} of {total_pages} from a Dutch government "
        "Woo-verzoek (freedom of information request) document dump.\n\n"
        f"Previous page was: {prev_summary}\n\n"
        "Classify this page. Respond ONLY with a JSON object, no explanation, no markdown:\n\n"
        "{\n"
        '  "page_type": "<email | nota | rapport | brief | bijlage | besluit | onbekend>",\n'
        '  "is_new_document": <true or false>,\n'
        '  "confidence": <0.0 to 1.0>,\n'
        '  "signals": ["<list of visual/textual clues you used>"],\n'
        '  "date": "<ISO date string if visible, else null>",\n'
        '  "sender": "<sender name or email if visible, else null>",\n'
        '  "recipient": "<recipient if visible, else null>",\n'
        '  "subject": "<subject line or document title if visible, else null>",\n'
        '  "doc_id_hint": "<any kenmerk, reference number, or document ID visible, else null>"\n'
        "}\n\n"
        "Rules:\n"
        "- is_new_document=true if this page clearly starts a new document "
        "(new email header, new document title, page 1 stamp visible, strong topic/format break)\n"
        "- is_new_document=false if this page continues the previous document\n"
        "- confidence should reflect how certain you are about the boundary decision\n"
        "- For the first page, always set is_new_document=true"
    )


# ---------------------------------------------------------------------------
# Ollama API call (with retry)
# ---------------------------------------------------------------------------

def _safe_fallback(page_index: int) -> dict:
    """Return a zero-confidence fallback result dict for *page_index*."""
    return {
        "page_index":      page_index,
        "skipped":         False,
        "page_type":       "onbekend",
        "is_new_document": False,
        "confidence":      0.0,
        "signals":         [],
        "date":            None,
        "sender":          None,
        "recipient":       None,
        "subject":         None,
        "doc_id_hint":     None,
    }


def _call_ollama(image_b64: str, prompt: str, model: str, page_index: int) -> dict:
    """Send image + prompt to Ollama; parse JSON response. Retries up to _MAX_RETRIES."""
    payload = {
        "model":   model,
        "prompt":  prompt,
        "images":  [image_b64],
        "stream":  False,
        "options": {"temperature": _TEMP, "num_ctx": _NUM_CTX},
    }
    last_raw = ""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            raw      = resp.json().get("response", "").strip()
            last_raw = raw
            # Strip markdown code fences some models add
            cleaned  = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            parsed   = json.loads(cleaned)
            result   = _safe_fallback(page_index)
            result.update(parsed)
            result["page_index"] = page_index  # override in case model returned wrong value
            result["skipped"]    = False
            return result
        except requests.exceptions.ConnectionError:
            raise OllamaNotAvailableError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull the model: ollama pull qwen2.5vl:7b"
            )
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                continue
            print(f"[vlm] JSON parse error on page {page_index + 1}: {exc}")
            print(f"[vlm]   Raw response: {repr(last_raw[:200])}")
            break
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                continue
            print(f"[vlm] API error on page {page_index + 1}: {exc}")
            break
    return _safe_fallback(page_index)


# ---------------------------------------------------------------------------
# Page selection (30-minute budget strategy)
# ---------------------------------------------------------------------------

def _select_pages(
    page_data: list[dict], total: int
) -> tuple[set[int], int, int]:
    """
    Choose which page indices to process based on the budget strategy.

    For ≤120 pages: all pages.
    For >120 pages: uncertain pages (no stamp), first page of each stamped
    document group, and any page with within_doc_page==1.

    Returns (selected_indices, n_uncertain, n_doc_starts).
    Note: the two sets are disjoint (uncertain requires code=None, doc_starts
    requires code!=None), so len(selected) == n_uncertain + n_doc_starts.
    """
    if total <= 120:
        return set(range(total)), total, 0

    uncertain_set:  set[int] = set()
    doc_start_set:  set[int] = set()
    prev_code: str | None = None

    for i, p in enumerate(page_data):
        code = p.get("detected_code")
        wpn  = p.get("within_doc_page")

        if code is None:
            uncertain_set.add(i)
        else:
            if code != prev_code:
                doc_start_set.add(i)
            prev_code = code
            if wpn == 1:
                doc_start_set.add(i)

    selected = uncertain_set | doc_start_set
    return selected, len(uncertain_set), len(doc_start_set)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_pages(
    images:    list[Image.Image],
    page_data: list[dict],
    model:     str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Run Qwen2.5-VL on selected page images.

    Selection follows the 30-minute budget strategy (see _select_pages).
    Processes pages sequentially with _PAGE_SLEEP delay between calls.

    Returns one dict per page (in page order):
    - Skipped pages:   {"page_index": i, "skipped": True,  "confidence": None}
    - Processed pages: {page_index, skipped=False, page_type, is_new_document,
                        confidence, signals, date, sender, recipient, subject,
                        doc_id_hint}

    Raises OllamaNotAvailableError if Ollama cannot be reached.
    """
    total = len(images)
    if total == 0:
        return []

    selected, n_uncertain, n_doc_starts = _select_pages(page_data, total)

    print(
        f"[vlm] {total} pages total — running VLM on {len(selected)} pages"
        f" (uncertain: {n_uncertain}, doc-starts: {n_doc_starts})"
    )
    est_secs = len(selected) * 15
    if est_secs >= 60:
        print(f"[vlm] Estimated time: ~{est_secs // 60} min {est_secs % 60} s")
    else:
        print(f"[vlm] Estimated time: ~{est_secs} s")

    results:      list[dict] = []
    prev_summary: str        = "none (this is the first page)"

    for i, image in enumerate(images):
        if i not in selected:
            results.append({"page_index": i, "skipped": True, "confidence": None})
            continue

        page_num = i + 1
        prompt   = _build_prompt(page_num, total, prev_summary)

        print(f"[vlm] Page {page_num}/{total}…", end=" ", flush=True)
        image_b64 = _encode_image(image)
        result    = _call_ollama(image_b64, prompt, model, i)

        conf   = result.get("confidence") or 0.0
        ptype  = result.get("page_type",       "onbekend")
        is_new = result.get("is_new_document", False)
        print(f"type={ptype}  new_doc={is_new}  conf={conf:.2f}")

        prev_summary = f"{ptype} (new_doc={is_new}, confidence={conf:.2f})"
        results.append(result)

        if i < total - 1:
            time.sleep(_PAGE_SLEEP)

    return results
