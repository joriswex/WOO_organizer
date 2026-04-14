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
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

OPENAI_MODEL    = "gpt-4o-2024-11-20"       # pinned snapshot for reproducibility
_BOUNDARY_MODEL = "gpt-4o-mini-2024-07-18"  # text-only boundary pass — no vision needed, much cheaper
_RENDER_DPI     = 200
_MAX_IMG_PX     = 1568      # GPT-4o "high" detail works best ≤ 2048px; 1568 is optimal tile size
_MAX_RETRIES    = 3
_MAX_WORKERS    = 10        # concurrent GPT-4o API calls
_STAMP_THRESHOLD = 0.30     # min fraction of pages with a detected stamp to use forward-fill
_NOTA_LIKE       = frozenset({"Nota", "Brief", "Report", "Vergadernotulen"})  # categories where same-code sub-splitting is safe

# WOO redaction code pattern — same as pipeline_ocr.py
_REDACTION_RE = re.compile(r"5\.[12]\.[1-9][a-z]{0,2}", re.IGNORECASE)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are analyzing pages from Dutch government WOO (Wet Open Overheid) disclosure dossiers. "
    "These documents were released under Dutch freedom-of-information law and may contain any type "
    "of internal government communication.\n"
    "Key context you must apply:\n"
    "- IMPORTANT: not all WOO dossiers use inventory stamps. When stamps ARE consistently present, "
    "they are the PRIMARY signal for document boundaries — a stamp code change between pages means "
    "a new document starts. Only fall back to content signals (headers, page counters, document "
    "type changes) when no stamps are visible.\n"
    "- Stamp codes are typically stamped in a corner or at the bottom-centre/top-centre of each "
    "page. Common formats: 4-digit (e.g. 0143), 6-digit, 7-digit barcode (760XXXD where XXX is "
    "the 3-digit document number), or a 'Doc XX' / 'Docnr XX' text label — space optional "
    "(e.g. 'Doc 5', 'Doc5', 'Docnr 23', 'Docnr23') — first page of each document only. "
    "All pages of the same document share the same code.\n"
    "- Some pages also show an incrementing per-page sequence counter (e.g. 00001, 00002, 00003 "
    "or 0001, 0002, 0003 — unique per page, increases by 1 each page across the entire dossier). "
    "These are NOT document codes; ignore them even if they look like a 4-digit leading-zero code.\n"
    "- Sensitive content is gelakt (blacked out) with redaction bars (some are black bars, other times white boxes or otherwise). The applicable WOO legal ground "
    "is printed in small text inside or next to each black box, in the format 5.1.X or 5.2.X "
    "(e.g. '5.1.2e', '5.1.1', '5.2.1'). These are articles of the Wet Open Overheid. "
    "WOO redaction articles are NOT document stamp codes — never return them as doc_code, "
    "even if printed near a margin.\n"
    "- Stamp codes are physically separate elements printed in a corner margin or centre strip, "
    "visually distinct from the document's own text. Numbers found inside the document body — "
    "including reference lines ('Kenmerk: 0230'), first lines of body text, case numbers, or "
    "any number embedded in a sentence — are NOT stamp codes; return null for doc_code in those cases.\n"
    "- The dossier sometimes starts with an Inventarislijst: a table listing all documents with "
    "their codes, titles, page counts, and the WOO decision (openbaar/deels openbaar/niet openbaar).\n"
    "- Document types in order of frequency: internal emails, forwarded email threads, memos "
    "(nota's), formal letters (brieven), reports, meeting minutes, chat exports, and legal decisions.\n"
    "Respond ONLY with a valid JSON object matching the schema in the user message. "
    "Do not include markdown code fences or any text outside the JSON object."
)

_PAGE_PROMPT = """\
Analyze this Dutch government document page and return a JSON object with exactly these fields:

{
  "is_new_document": <true if this is clearly the first page of a new document>,
  "doc_code": "<stamp code from a corner or edge, reported exactly as printed (e.g. '0143', '760143') — or null if not visible>",
  "within_doc_page": <integer — page number within this document, e.g. 1, 2, 3 — or null>,
  "category": "<Email | Chat | Nota | Report | Brief | Timeline | Vergadernotulen | Inventarislijst | Other>",
  "doc_subtype": "<email | chat_sms | nota | brief | factuur | besluit | kamerbrief | vergaderverslag | persbericht | rapport | other>",
  "has_redactions": <true if black boxes or censored areas are visible>,
  "doc_date": "<primary date as YYYY-MM-DDTHH:MM if a time is visible, else YYYY-MM-DD, or null — see METADATA RULES>",
  "doc_sender": "<author or sender as a plain name string, or null>",
  "email_start": <true if a new email header block starts on this page, else false>,
  "email_from": "<sender from Van:/From: field, or null>",
  "email_to": "<recipient(s) from Aan:/To: field, or null>",
  "email_cc": "<CC field, or null>",
  "email_subject": "<subject from Onderwerp:/Subject: field, or null>",
  "email_date": "<sent date as YYYY-MM-DDTHH:MM if a time is visible, else YYYY-MM-DD, or null>",
  "chat_name": "<name of the chat group or contact if this is a Chat page, else null>",
  "chat_messages": <array of message objects if this is a Chat page, else []>,
  "text": "<all readable text from the page — see TEXT EXTRACTION RULES>"
}

TEXT EXTRACTION RULES:
- Extract ALL text in reading order, preserving paragraph and section structure.
- For email documents, preserve header fields each on their own line:
    Van: ...   Aan: ...   CC: ...   Onderwerp: ...   Datum: ...   Verzonden: ...
- For tables (e.g. Inventarislijst), reproduce rows as tab- or pipe-separated text so the
  structure is recoverable. Include all column values including codes and page counts.
- For page headers/footers (logo text, document reference numbers, "VERTROUWELIJK" stamps):
  include them at the top or bottom of the text field respectively.
- For gelakte email addresses where only the domain remains (black box before @domain.nl):
    write: [GELAKT]@domain.nl
- For gelakte sections (solid black rectangles / censored areas):
    If a WOO article code (e.g. "5.1.2e") is printed next to the box:  write [GELAKT: 5.1.2e]
    If no article code is visible:                                       write [GELAKT]
- For media in chat exports ([Foto], [Video], [Bestand], [Sticker], [Audio]):
    include the placeholder exactly as printed, e.g. "[Foto]" or "[Video weggelaten]".
- If the page is blank or contains only a stamp/page number with no readable content:
    set text to "" (empty string).
- Use blank lines to separate paragraphs and sections.

DOCUMENT BOUNDARY RULES:
- is_new_document = true ONLY when ONE OR MORE of these conditions hold at the very top of the page:
    * Footer or stamp reads "Pagina 1 van N" or "1/N"
    * A nota or letter heading: document title + date + reference number (kenmerk/zaaknummer) at
      lines 1–3, in a clear Nota / Brief / Memorandum / Rapport layout — this is the strongest
      signal; a new heading block with fresh metadata always means a new document
    * A fresh original email header (Van: + Aan: + Onderwerp:) at lines 1–3, but ONLY when it is
      clearly a new standalone email — NOT when it is a RE: or FW: reply/forward continuing an
      existing thread visible earlier in the document
    * The page shows a new Inventarislijst table header
    * A forwarded-message block ("-----Doorgestuurd bericht-----" / "-----Forwarded Message-----")
      appears at the very top, clearly starting a new forwarded document
    * A "Doc XX" or "Docnr XX" label is visible in a corner (these appear ONLY on the first page
      of each document — continuation pages have no such label)
- is_new_document = false in ALL of the following cases:
    * within_doc_page is 2 or higher
    * The page starts with a salutation like "Geachte" or "Beste" but has NO header block above it
      (this is the continuation of a letter or email body, not a new document)
    * The page starts mid-sentence, mid-paragraph, or mid-list (clear continuation)
    * Quoted/forwarded email headers appear only deep in the body, not at the very top
    * No stamp is visible and the preceding page had a stamp (continuation is the default)
    * The page is blank or contains only a page separator / cover sheet
- doc_code: check ALL four corners AND the bottom-centre/top-centre for a stamp code.
    * Common formats: 4-digit (e.g. "0143"), 6-digit, 7-digit barcode, or "Doc XX" / "Docnr XX"
      label — space between "Doc"/"Docnr" and the number is optional (e.g. "Doc 23", "Doc23",
      "Docnr 143", "Docnr143" are all the same). Return the number zero-padded to 4 digits
      (e.g. "Doc 23" → "0023", "Docnr143" → "0143").
    * Year-like numbers (1900–2099) are NOT document codes — ignore them.
    * Incrementing per-page sequence counters (e.g. 00001, 00002, 00003 — every page gets a
      different, incrementing number) are NOT document codes — ignore them.
    * WOO redaction articles (e.g. "5.1.2", "5.1.2e", "5.2.1") are NOT document codes — ignore
      them even if they appear near a page margin (they are printed next to redaction bars).
    * Stamp codes are PHYSICALLY DISTINCT elements — printed as a discrete stamp or label in a
      corner margin, clearly separate from the document's own text. They are NOT:
        - Numbers in a document reference line (e.g. "Kenmerk: 0230", "Ons kenmerk 0230")
        - Numbers appearing as the first word or first line of body text
        - Case numbers, file numbers, or any number embedded in a sentence or header field
        - Numbers that appear on only one or two pages of a multi-page document
      If a number appears anywhere in the body of the document (even at the top), return null.
    * Document codes repeat: all pages of the same document share the same stamp code.
      "Doc XX" labels are an exception — they appear ONLY on the first page; return null for
      continuation pages (the pipeline will forward-fill the code automatically).
- within_doc_page: return X if a stamp or footer explicitly reads "Pagina X van N" or "pag. X".
  Return null if no explicit page-within-document counter is visible.

CATEGORY RULES (pick the best match; when ambiguous prefer the more specific type):
- Email:          Van/Aan/Onderwerp header block; From/To/Subject block; Outlook meeting invite
                  (Required Attendees, Start Date/Time); reply body with quoted headers below
- Chat:           Screenshot or print of a messaging app (WhatsApp, Signal, Teams, SMS, iMessage)
                  showing speech bubbles or a forensic chat export (sms.db, Native Messages)
- Nota:           Internal memo with "NOTA", "Memorandum", or "Briefing" heading; "Aan:", "Van:", "Betreft:"
- Brief:          Formal letter with address block, date, and salutation ("Geachte", "Beste", "Dear")
                  NOTE: a Brief addressed to a minister or parliament member that also has email-like
                  header fields should still be classified as Brief, not Email.
- Report:         Multi-page document with chapter headings, "Rapport", "Onderzoek", "Analyse"
- Timeline:       Chronological list of events or dates, "Tijdlijn", "Chronologisch overzicht"
- Vergadernotulen: Meeting minutes or agenda; "Verslag", "Notulen", "Agenda", "Aanwezig:"
- Inventarislijst: Table inventorying WOO dossier documents — columns: code, title, pages, decision
- Other:          Does not fit any of the above

DOC_SUBTYPE RULES — return exactly one of:
  email | chat_sms | nota | brief | factuur | besluit | kamerbrief | vergaderverslag | persbericht | rapport | other
- email:          Van:/Aan:/Onderwerp: or From:/To:/Subject: header fields
- chat_sms:       Chat bubbles, forensic metadata ("FG ProMax", "sms.db", "Status: Sent/Read")
- nota:           "NOTA" heading with Aan:/Van:/Betreft: block
- brief:          Address block + "Geachte"/"Beste"/"Dear" salutation
- factuur:        "Factuur", "BTW", "Bedrag", invoice/order numbers, payment totals
- besluit:        "Besluit", "ingevolge artikel", legal operative clauses, "overwegende dat"
- kamerbrief:     "Aan de Voorzitter van de Tweede Kamer" or "Aan de Voorzitter van de Eerste Kamer"
- vergaderverslag: "Verslag", "Notulen", "Aanwezig:", "Agenda", action-item lists
- persbericht:    Press-release heading, embargo line, "Persbericht", public statement format
- rapport:        Chapter headings/numbered sections spanning multiple pages, "Rapport", "Onderzoek"
- other:          Use only when none of the signals above apply

CHAT PAGE RULES (only when category = "Chat"):
- chat_name: the group or contact name shown at the top of the screen (e.g. "Bezoek praktische zaken").
  If the name is redacted, write "[GELAKT]".
- Extract every visible message into chat_messages (array, top-to-bottom order):
  {
    "sender_position": "left" | "right",   // left = incoming, right = outgoing (device owner)
    "sender_label": "<name or redaction code above the bubble; '[GELAKT]' if blacked out>",
    "timestamp": "<HH:MM visible on the message, or null>",
    "is_system_message": <true for date separators, join/leave events, missed-call notices>,
    "content": "<full message text; use [GELAKT] for blacked-out words; media as [Foto]/[Video]/[Audio]/[Bestand]>"
  }
- System messages (date separators like "Vandaag", "Gisteren", "15 oktober 2024"; join/leave
  notifications; "Berichten en oproepen zijn end-to-end versleuteld") should be included with
  is_system_message = true, sender_position = "left", sender_label = null.
- If a sender name is consistently replaced by the same redaction code (e.g. "5.1.2e"), use
  that code as sender_label across all that person's messages so they can be tracked.
- text field: plain concatenation of all non-system message contents (for search/sorting).

METADATA RULES (apply to ALL document types, including emails):
- doc_date: the primary date this document was written, sent, or issued.
    * Sources: letterhead date, "Datum:", "Date:", "Verzonden:", "Sent:", "Start Date/Time:"
    * Dutch date formats to parse:
        "3 januari 2024"          → "2024-01-03"
        "15 okt. 2024"            → "2024-10-15"   (jan feb mrt/maa apr mei jun jul aug sep okt nov dec)
        "ma 06-03-2024 14:32"     → "2024-03-06T14:32"  (strip day abbreviation)
        "Verzonden: di 14:32"     combined with a visible date → use the date + "T14:32"
        "06/03/2024 14:32:05"     → "2024-03-06T14:32"  (strip seconds)
    * If a time is visible: return "YYYY-MM-DDTHH:MM" (strip seconds if present).
    * For email threads: use the date/time of the most recent (topmost) email on this page.
    * Return null ONLY if no date is visible anywhere on the page.
- doc_sender: name only — no email addresses, no angle brackets.
    * Email: Van:/From: name
    * "Verzonden namens [Name]" or "On behalf of [Name]": use the delegated name
    * Nota/Brief: name or team from signature or "Van:" header line
    * Report: organisation or author from title page
    * Return null if not determinable.

EMAIL HEADER RULES (only when category = "Email"):
- email_start = true when this page begins a NEW email. Signals:
    * Fresh Van:/From: + Aan:/To: + Onderwerp: block at lines 1–5
    * Outlook meeting invite: From: + Required Attendees: + Subject: + Start Date/Time:
    * Reply body starting with salutation/text before any quoted headers on this page
      (e.g. page begins with "Hoi," then shows "-----Oorspronkelijk bericht-----" below)
    * First page of a forwarded email marked "FW:" or "Fwd:" in the subject
- email_start = false if the page is a continuation of an email body or quoted thread.
- email_from: name only; strip email addresses.
    * "Verzonden namens [Name]" → use the delegated name
    * Partially redacted (e.g. "[GELAKT]@minbuza.nl"): write "[GELAKT]"
- email_to, email_cc: names only, comma-separated for multiple recipients. Null if absent.
- email_subject: include FW:/Re:/Fwd: prefixes as-is. "[GELAKT]" if fully redacted.
- email_date: parse Datum:/Verzonden:/Sent:/Date:/Start Date/Time: → YYYY-MM-DD or YYYY-MM-DDTHH:MM.
    Strip day abbreviations and seconds. Null if not visible or not parseable.
- If multiple emails start on the same page: report only the FIRST one's header fields.
- Non-email pages: email_start = false, all email_* fields = null.
"""


# ── Pass-0: dossier profile ───────────────────────────────────────────────────
# Profile is derived from a small pilot of real pass-1 calls; no separate prompt needed.

_PROFILE_N_SAMPLES = 8    # pilot pages run through full pass-1 to determine stamp format


def _profile_dossier(images: list, client) -> tuple[dict, dict[int, dict]]:
    """
    Pass 0: run a small pilot of real pass-1 calls on a spread of pages, then
    determine the stamp format from the returned doc_code values in Python.

    This is more reliable than asking GPT-4o to compare images directly, because
    Python can count exact code repetitions — distinguishing a repeating document
    code from an incrementing per-page counter whose digits look visually similar.

    The pilot page results are returned so load_pdf_vlm() can slot them directly
    into the main pass-1 loop, avoiding double-processing of those pages.

    Returns:
        profile       — dict: stamp_format, stamp_example, notes
        pilot_results — {page_index: raw_gpt4o_result} for reuse in the main loop
    """
    total = len(images)
    n     = min(_PROFILE_N_SAMPLES, total)

    if total <= n:
        indices = list(range(total))
    else:
        # Evenly distributed + a cluster of 4 consecutive pages around 35% of the doc.
        # The cluster ensures adjacent pages are seen together so same-document code
        # repetition is visible even in dense dossiers with many short documents.
        n_cluster     = min(4, n // 2)
        n_spread      = n - n_cluster
        cluster_start = max(0, round(total * 0.35) - n_cluster // 2)
        cluster_idx   = list(range(cluster_start, min(total, cluster_start + n_cluster)))
        step          = (total - 1) / max(n_spread - 1, 1)
        spread_idx    = [round(i * step) for i in range(n_spread)]
        indices       = sorted(set(cluster_idx) | set(spread_idx))

    print(f"[gpt4o] Sampling {len(indices)} pages to detect stamp format...")

    # Run pilot pages through the real pass-1 call (no hint yet — generic prompt)
    pilot_results: dict[int, dict] = {}
    for idx in indices:
        b64 = _encode_image(images[idx])
        pilot_results[idx] = _call_gpt4o(b64, client, idx, pdf_page_num=idx + 1)

    # ── Analyse returned doc_codes in Python ──────────────────────────────────
    raw_strs  = [str(pilot_results[i].get("doc_code") or "") for i in indices]
    norm_codes = [_normalise_doc_code(pilot_results[i].get("doc_code")) for i in indices]
    valid      = [c for c in norm_codes if c]
    code_freq  = Counter(valid)

    # DocnrXX: GPT-4o normalises "Docnr5" → "0005" BUT the raw string still contains
    # "docnr" / "doc" — check raw strings before normalisation.
    has_docnr = any(re.search(r"docnr?\s*\d", s, re.IGNORECASE) for s in raw_strs)

    if has_docnr:
        fmt     = "docnr"
        example = next((s for s in raw_strs
                        if re.search(r"docnr?\s*\d", s, re.IGNORECASE)), None)
        notes   = f"DocnrXX label detected in pilot (raw: {example!r})"

    elif not valid:
        fmt     = "none"
        example = None
        notes   = "No stamp codes returned by pass-1 on pilot pages"

    else:
        # Determine format from raw digit lengths (before normalisation collapses them).
        # 4-digit is the most common WOO stamp format — prefer it whenever any pilot
        # page returned a 4-digit code, even if longer codes are more frequent
        # (longer codes are often per-page counters that coexist with the real stamp).
        raw_digit_lens = [len(re.sub(r"\D", "", s)) for s in raw_strs if s]
        dominant_len   = Counter(raw_digit_lens).most_common(1)[0][0] if raw_digit_lens else 4
        if any(l == 4 for l in raw_digit_lens):
            fmt = "4-digit"
        elif dominant_len <= 6:
            fmt = "6-digit"
        else:
            fmt = "7-digit"

        # Prefer the most frequent normalised code as the example
        repeats = {c: cnt for c, cnt in code_freq.items() if cnt >= 2}
        example = max(repeats, key=lambda c: repeats[c]) if repeats else (valid[0] if valid else None)

        if repeats:
            top_code = max(repeats, key=lambda c: repeats[c])
            notes = (f"Code {top_code!r} seen {repeats[top_code]}× in pilot "
                     f"({len(set(valid))} distinct codes across {len(indices)} pages)")
        else:
            notes = (f"No repeats in pilot ({len(indices)} pages, "
                     f"{len(set(valid))} distinct codes) — format inferred from digit length")

    ex_str = f" (e.g. '{example}')" if example else ""
    print(f"[gpt4o] Stamp format detected: {fmt}{ex_str}")
    return {"stamp_format": fmt, "stamp_example": example, "notes": notes}, pilot_results


def _profile_to_hint(profile: dict) -> str:
    """
    Convert a pass-0 dossier profile into a concrete system-prompt addition
    that steers pass-1 toward the correct stamp format and location.
    Returns an empty string when no useful profile is available.
    """
    fmt  = profile.get("stamp_format", "")
    loc  = profile.get("stamp_location", "")
    ex   = profile.get("stamp_example")
    notes = profile.get("notes", "")

    loc_str = f" in the {loc}" if loc not in ("", "unknown", "multiple") else ""
    ex_str  = f' (example seen: "{ex}")' if ex else ""
    note_str = f" Note: {notes}" if notes else ""

    if fmt == "none":
        return (
            "DOSSIER-SPECIFIC OVERRIDE: This dossier contains NO document stamp codes. "
            "Always return null for doc_code on every page. "
            "Use content signals only (email headers, page counters, document headings) "
            "for is_new_document."
        )
    if fmt == "docnr":
        loc_hint = f" (typically seen {loc_str.strip()}, but check ALL four corners)" if loc_str else " (check ALL four corners)"
        return (
            f"DOSSIER-SPECIFIC OVERRIDE: This dossier uses 'DocnrXX' / 'DocXX' text labels"
            f"{loc_hint} to identify documents{ex_str}. "
            "Space between 'Docnr'/'Doc' and the number is optional. "
            "These labels appear ONLY on the FIRST page of each document; "
            "continuation pages have no label — return null for doc_code on those pages. "
            "is_new_document = true when a 'Doc'/'Docnr' label is visible in any corner. "
            f"Ignore all other numbers for doc_code.{note_str}"
        )
    if fmt == "4-digit":
        return (
            f"DOSSIER-SPECIFIC OVERRIDE: This dossier uses 4-digit stamp codes"
            f"{loc_str}{ex_str}. "
            "Every page of the same document shares the same code; a code change signals a new document. "
            "For doc_code: look only for a 4-digit number with a leading zero stamped in the "
            f"corners or top/bottom strip — not in the body text. "
            "If two 4-digit numbers appear on the same page, use the one that is shared by "
            "consecutive pages (the stable doc code), NOT the one that increments by 1 each "
            f"page across the dossier (a per-page counter).{note_str}"
        )
    if fmt == "6-digit":
        return (
            f"DOSSIER-SPECIFIC OVERRIDE: This dossier uses 6-digit stamp codes"
            f"{loc_str}{ex_str}. "
            f"Every page of the same document shares the same code.{note_str}"
        )
    if fmt in ("7-digit", "barcode-7digit"):
        return (
            f"DOSSIER-SPECIFIC OVERRIDE: This dossier uses 7-digit stamp codes"
            f"{loc_str}{ex_str}. "
            "Report the full 7-digit code exactly as it appears — do not shorten or reformat it. "
            "Every page of the same document shares the same 7-digit code; "
            f"a code change means a new document starts.{note_str}"
        )
    return ""   # unknown format — fall back to the generic prompt


# ── Raster stamp survey (pass-0b, runs before GPT-4o) ────────────────────────

def _raster_stamp_survey(images: list, page_offset: int = 0) -> dict:
    """
    Cheap Tesseract-only pre-scan of all pages to determine whether the dossier
    uses stamps, what the code range is, and where jumps signal boundaries.

    Runs before any GPT-4o call so the result can:
      - Skip the GPT-4o pilot entirely for stampless dossiers (saves API cost)
      - Inject a per-page code table into the pass-1 system prompt
      - Replace _ocr_stamp_fallback as the authoritative raster source

    Consistency rules for "has_stamps = True":
      - At least 2 distinct valid codes
      - Coverage >= 10% of pages
      - Codes are generally monotonically non-decreasing (>= 60% of consecutive
        pairs with a valid code are non-decreasing) — noise is tolerated but
        random churn is not
    """
    from pipeline_ocr import _find_doc_code_raster, _DOC_CODE_RASTER_REGIONS

    n = len(images)
    raw:          list[str | None] = [None] * n
    raw_regions:  list[str | None] = [None] * n   # which region produced each code
    docnr_flags:  list[bool]       = [False] * n

    def _scan(args: tuple[int, object]) -> tuple[int, str | None, str | None, bool]:
        idx, img = args
        result = _find_doc_code_raster(img, return_location=True)
        if result:
            code, region = result
            return idx, _normalise_doc_code(code), region, False
        # Check for DocNr text labels (e.g. "Docnr26") — no leading zero, so the
        # numeric regex in _find_doc_code_raster cannot match them.  A quick
        # Tesseract pass on the top strip is enough to detect the pattern.
        try:
            import pytesseract
            w, h = img.size
            top  = img.crop((0, 0, w, int(h * 0.15)))
            txt  = pytesseract.image_to_string(top, lang="nld+eng", config="--psm 6")
            if re.search(r"docnr?\s*\d+", txt, re.IGNORECASE):
                return idx, None, None, True
        except Exception:
            pass
        return idx, None, None, False

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for idx, code, region, is_docnr in ex.map(_scan, enumerate(images)):
            raw[idx]         = code
            raw_regions[idx] = region
            docnr_flags[idx] = is_docnr

    # ── Dominant-region lock-in ───────────────────────────────────────────────
    # Stamps in a dossier are always at the same position.  Count which region
    # produced the most hits; if one region has a clear majority (>50%), restrict
    # a second pass to that region only.  This prevents a different region from
    # producing spurious hits on pages where the dominant region is blank (e.g.
    # btm-left picking up fragments of an 8-digit code when btm-middle is the
    # true stamp location).
    region_hits: dict[str, int] = {}
    for r in raw_regions:
        if r:
            region_hits[r] = region_hits.get(r, 0) + 1

    dominant_region: str | None = None
    if region_hits:
        total_hits = sum(region_hits.values())
        top_region, top_count = max(region_hits.items(), key=lambda x: x[1])
        if top_count / total_hits > 0.50:
            dominant_region = top_region

    if dominant_region:
        dominant_region_spec = [r for r in _DOC_CODE_RASTER_REGIONS if r[0] == dominant_region]
        if dominant_region_spec:
            # Pages whose code came from a non-dominant region need re-scanning
            # restricted to the dominant region only (may return None if blank there).
            pages_to_rescan = [
                (idx, images[idx]) for idx in range(n)
                if raw_regions[idx] and raw_regions[idx] != dominant_region
            ]
            def _rescan(args: tuple[int, object]) -> tuple[int, str | None]:
                idx, img = args
                result = _find_doc_code_raster(img, regions=dominant_region_spec)
                return idx, _normalise_doc_code(result) if result else None

            if pages_to_rescan:
                with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
                    for idx, corrected_code in ex.map(_rescan, pages_to_rescan):
                        raw[idx] = corrected_code   # None if stamp not at dominant loc
                print(f"[gpt4o] Dominant stamp region: {dominant_region!r} "
                      f"({top_count}/{total_hits} hits); "
                      f"re-scanned {len(pages_to_rescan)} page(s) restricted to that region.")

    docnr_pages      = sum(docnr_flags)
    has_docnr_labels = docnr_pages >= 2   # require ≥2 pages to avoid stray OCR hits

    valid_pages = [(i, c) for i, c in enumerate(raw) if c]
    coverage    = len(valid_pages) / n if n else 0.0
    distinct    = sorted({c for _, c in valid_pages})

    if len(distinct) < 2 or coverage < 0.10:
        return {
            "has_stamps":      False,
            "has_docnr_labels": has_docnr_labels,
            "coverage":        coverage,
            "codes":           raw,
            "distinct_count":  len(distinct),
            "range":           None,
            "jumps":           [],
        }

    # Monotonicity check — real stamps go up; noise bounces randomly
    int_vals = []
    for _, c in valid_pages:
        try:
            int_vals.append(int(c))
        except ValueError:
            pass

    if len(int_vals) >= 2:
        non_dec = sum(1 for a, b in zip(int_vals, int_vals[1:]) if b >= a)
        monotone_ratio = non_dec / (len(int_vals) - 1)
    else:
        monotone_ratio = 1.0

    if monotone_ratio < 0.60:
        return {
            "has_stamps":       False,
            "has_docnr_labels": has_docnr_labels,
            "coverage":         coverage,
            "codes":            raw,
            "distinct_count":   len(distinct),
            "range":            None,
            "jumps":            [],
        }

    # Page-counter detection: real doc codes repeat across consecutive pages.
    # A per-page sequence counter (0001, 0002, 0003...) has every code unique —
    # unique_ratio ≈ 1.0.  If virtually all codes are unique AND the sequence
    # increments by exactly 1 nearly every step, it is a page counter, not stamps.
    unique_ratio = len(distinct) / len(valid_pages) if valid_pages else 0.0
    if unique_ratio > 0.90 and len(int_vals) >= 4:
        step_1 = sum(1 for a, b in zip(int_vals, int_vals[1:]) if b - a == 1)
        step_1_ratio = step_1 / (len(int_vals) - 1)
        if step_1_ratio > 0.80:
            # The raster codes are a page counter, not document stamps.
            # If we know which region produced them, rescan every page excluding
            # that region — the real stamp may be at a different position.
            counter_region = dominant_region
            fallback_regions = (
                [r for r in _DOC_CODE_RASTER_REGIONS if r[0] != counter_region]
                if counter_region else None
            )
            raw2 = [None] * n
            if fallback_regions:
                def _rescan_excl(args: tuple[int, object]) -> tuple[int, str | None]:
                    idx, img = args
                    res = _find_doc_code_raster(img, regions=fallback_regions)
                    return idx, _normalise_doc_code(res) if res else None
                with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
                    for idx, code in ex.map(_rescan_excl, enumerate(images)):
                        raw2[idx] = code
                valid2   = [(i, c) for i, c in enumerate(raw2) if c]
                distinct2 = {c for _, c in valid2}
                # Accept the fallback scan if it found ≥2 distinct repeating codes
                if len(distinct2) >= 2 and len(valid2) > len(distinct2):
                    print(f"[gpt4o] Page counter detected in {counter_region!r}; "
                          f"fallback scan (excluding that region) found "
                          f"{len(distinct2)} distinct stamp code(s) — using those.")
                    # Replace raw with fallback results and recompute derived values.
                    raw        = raw2
                    valid_pages = valid2
                    distinct    = sorted(distinct2)
                    int_vals    = []
                    for _, c in valid_pages:
                        try:
                            int_vals.append(int(c))
                        except ValueError:
                            pass
                    dominant_region = counter_region  # keep for location hint
                    # Fall through to jump detection with corrected codes.
                else:
                    print(f"[gpt4o] Page counter detected in {counter_region!r}; "
                          f"fallback scan found no stable stamps — flagging stampless.")
                    return {
                        "has_stamps":            False,
                        "has_docnr_labels":      has_docnr_labels,
                        "coverage":              coverage,
                        "codes":                 [None] * n,
                        "distinct_count":        0,
                        "range":                 None,
                        "jumps":                 [],
                        "_page_counter_detected": True,
                        "_counter_region":        counter_region,
                    }
            else:
                return {
                    "has_stamps":            False,
                    "has_docnr_labels":      has_docnr_labels,
                    "coverage":              coverage,
                    "codes":                 [None] * n,
                    "distinct_count":        0,
                    "range":                 None,
                    "jumps":                 [],
                    "_page_counter_detected": True,
                    "_counter_region":        None,
                }

    # Detect jumps: build runs of consecutive identical codes, then look for
    # integer gaps > 1 between runs of length >= 2.
    # Single-page runs (singletons) are excluded from jump detection because
    # a single misread stamp surrounded by a different sequence is almost
    # always an OCR error, not a real document boundary.
    runs: list[tuple[str, int, int]] = []   # (code, first_page_idx, run_length)
    if valid_pages:
        run_code, run_start, run_len = valid_pages[0][1], valid_pages[0][0], 1
        for pg_idx, code in valid_pages[1:]:
            if code == run_code:
                run_len += 1
            else:
                runs.append((run_code, run_start, run_len))
                run_code, run_start, run_len = code, pg_idx, 1
        runs.append((run_code, run_start, run_len))

    sustained = [(code, start) for code, start, length in runs if length >= 2]

    # Only flag jumps that are large enough to be meaningful — small gaps (≤ 5)
    # are almost always single-page documents whose stamps the raster OCR read as
    # singletons, not real skips in the sequence.
    _MIN_JUMP_GAP = 6
    jumps: list[dict] = []
    for i in range(1, len(sustained)):
        prev_code, _         = sustained[i - 1]
        cur_code,  cur_start = sustained[i]
        try:
            gap = int(cur_code) - int(prev_code)
        except ValueError:
            continue
        if gap > _MIN_JUMP_GAP:
            jumps.append({
                "page":      cur_start + page_offset,
                "from_code": prev_code,
                "to_code":   cur_code,
                "gap":       gap,
            })

    # Derive range from sustained runs only — singleton outliers (OCR errors on
    # a single page) can produce spurious high/low codes that distort the range.
    sustained_ints = [int(c) for c, _ in sustained if c.isdigit()]
    if sustained_ints:
        min_code = f"{min(sustained_ints):04d}"
        max_code = f"{max(sustained_ints):04d}"
    else:
        all_ints = [int(c) for c in distinct if c.isdigit()]
        min_code = f"{min(all_ints):04d}" if all_ints else None
        max_code = f"{max(all_ints):04d}" if all_ints else None

    return {
        "has_stamps":        True,
        "has_docnr_labels":  has_docnr_labels,
        "coverage":          coverage,
        "codes":             raw,
        "distinct_count":    len(distinct),
        "range":             (min_code, max_code) if min_code else None,
        "jumps":             jumps,
        "dominant_region":   dominant_region,
    }


def _raster_survey_to_hint(survey: dict, page_offset: int = 0) -> str:
    """Convert raster stamp survey to a system-prompt addition for GPT-4o pass-1."""
    codes = survey.get("codes", [])

    if not survey.get("has_stamps"):
        cov       = survey.get("coverage", 0)
        n         = survey.get("distinct_count", 0)
        has_docnr = survey.get("has_docnr_labels", False)
        if has_docnr:
            # Raster Tesseract detected "DocNrXX" text labels on ≥2 pages.
            # The numeric regex cannot match them (no leading zero), but the
            # pilot confirmed their presence.
            return (
                f"RASTER PRE-SCAN: 'DocNr' / 'Doc' text labels detected on ≥2 pages "
                f"({len(codes)} pages total). "
                "These labels have no leading zero (e.g. 'Docnr26'), so numeric stamp patterns "
                "do not apply. Look for 'DocNrXX' or 'DocXX' text in page corners — "
                "it appears ONLY on the first page of each document."
            )
        if survey.get("_page_counter_detected"):
            _REGION_LABEL = {
                "top-right":  "top-right corner",
                "top-left":   "top-left corner",
                "btm-right":  "bottom-right corner",
                "btm-middle": "bottom-centre strip",
                "btm-left":   "bottom-left corner",
                "top-full":   "top strip",
            }
            counter_region = survey.get("_counter_region")
            counter_loc = (f" in the {_REGION_LABEL.get(counter_region, counter_region)}"
                           if counter_region else "")
            return (
                f"RASTER PRE-SCAN: A monotonically incrementing 4-digit sequence was found"
                f"{counter_loc} ({len(codes)} pages). "
                f"This is a per-page counter, NOT a document stamp code. "
                f"Ignore any 4-digit number{counter_loc} that increments by 1 each page. "
                "Look for a separate 4-digit stamp that stays the same across consecutive pages "
                "of the same document — that is the doc_code. "
                "Return null if no such stable stamp is visible."
            )
        if cov <= 0.05:
            # Near-zero coverage — genuinely stampless
            return (
                f"RASTER PRE-SCAN: No stamp codes found across {len(codes)} pages "
                f"({cov:.0%} coverage). "
                "This appears to be a stampless dossier — return null for doc_code on every page "
                "unless a stamp is unambiguously visible in a page corner."
            )
        else:
            # Some fragments detected but no consistent sequence —
            # could be DocNr-format (first-page-only labels) or sparse stamps
            return (
                f"RASTER PRE-SCAN: Sparse or inconsistent stamp codes "
                f"({cov:.0%} coverage, {n} distinct value(s) — no consistent sequence). "
                "May be a DocNr-format dossier where stamps appear only on the first page "
                "of each document. Check all four corners carefully for 'DocNr XX', "
                "'Doc XX', or a 4-digit stamp code."
            )

    lo, hi          = survey["range"]
    cov             = survey["coverage"]
    n_dist          = survey["distinct_count"]
    jumps           = survey.get("jumps", [])
    dominant_region = survey.get("dominant_region")

    # Compact per-page table: only pages that have a detected code
    entries = [
        f"{i + page_offset + 1}:{c}"
        for i, c in enumerate(codes) if c
    ]
    # Keep the table under ~500 chars to avoid bloating the system prompt
    table = "  " + "  ".join(entries)
    if len(table) > 500:
        shown  = []
        length = 2
        for e in entries:
            if length + len(e) + 2 > 497:
                shown.append("...")
                break
            shown.append(e)
            length += len(e) + 2
        table = "  " + "  ".join(shown)

    jump_block = ""
    if jumps:
        lines = [
            f"  pg {j['page'] + 1}: {j['from_code']} → {j['to_code']} (gap {j['gap']})"
            for j in jumps[:15]
        ]
        jump_block = (
            "\nStamp sequence jumps (likely document boundaries where codes skipped):\n"
            + "\n".join(lines)
        )

    loc_hint = ""
    if dominant_region:
        # Map internal region names to human-readable descriptions for GPT-4o
        _REGION_LABEL = {
            "top-right":  "top-right corner",
            "top-left":   "top-left corner",
            "btm-right":  "bottom-right corner",
            "btm-middle": "bottom-centre strip",
            "btm-left":   "bottom-left corner",
            "top-full":   "top strip",
        }
        label = _REGION_LABEL.get(dominant_region, dominant_region)
        loc_hint = (
            f"\nSTAMP LOCATION: All stamps in this dossier are in the {label}. "
            "Look ONLY there for doc_code — ignore any 4-digit numbers in other positions."
        )

    return (
        f"RASTER PRE-SCAN: Stamps on {cov:.0%} of pages, "
        f"{n_dist} distinct codes, range {lo}–{hi}.\n"
        f"Per-page raster codes (pdf_page: code):\n{table}"
        f"{jump_block}"
        f"{loc_hint}\n"
        "Use these as a cross-check: if your visual reading differs from the raster hint, "
        "prefer what you see in the image — but treat the raster code as a strong prior."
    )


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
    "vergadernotulen":  "Vergadernotulen",
    "inventarislijst":  "Inventarislijst",
    "Inventarislijst":  "Inventarislijst",
    "other":            "Other",
    # Canonical pass-through (already normalised, e.g. loaded from cache)
    "E-mail":           "E-mail",
    "Nota":             "Nota",
    "Report":           "Report",
    "Brief":            "Brief",
    "Timeline":         "Timeline",
    "Vergadernotulen":  "Vergadernotulen",
    "Other":            "Other",
}

_VALID_DOC_SUBTYPES: set[str] = {
    "email",
    "chat_sms",
    "nota",
    "brief",
    "factuur",
    "besluit",
    "kamerbrief",
    "vergaderverslag",
    "persbericht",
    "rapport",
    "other",
}

_CATEGORY_TO_SUBTYPE: dict[str, str] = {
    "E-mail": "email",
    "Chat": "chat_sms",
    "Nota": "nota",
    "Brief": "brief",
    "Report": "rapport",
    "Vergadernotulen": "vergaderverslag",
    "Timeline": "other",
    "Inventarislijst": "other",
    "Other": "other",
}


def _normalise_category(raw: str) -> str:
    return _GPT4O_TO_CATEGORY.get(raw.strip(), "Other")


def _normalise_doc_subtype(raw: str | None, category: str) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if value in _VALID_DOC_SUBTYPES:
        return value
    return _CATEGORY_TO_SUBTYPE.get(category, "other")


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


# ── JSON repair ───────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> dict:
    """
    Repair common GPT JSON issues: trailing commas and truncation.
    - Strips trailing commas before } or ] (e.g. {"a": 1,} → {"a": 1})
    - Closes open string literals and unclosed brackets left by max_tokens truncation.
    """
    import re as _re
    s = raw.rstrip()
    # Remove trailing commas before closing brace/bracket (invalid but GPT does it occasionally)
    s = _re.sub(r",\s*([}\]])", r"\1", s)

    # Determine whether we ended inside a string literal
    in_string = False
    escaped   = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        s += '"'   # close the open string

    # Count unclosed brackets/braces (ignoring string contents)
    stack     = []
    in_string = False
    escaped   = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()

    s += "".join(reversed(stack))
    return json.loads(s)


# ── GPT-4o call ───────────────────────────────────────────────────────────────

def _call_gpt4o(image_b64: str, client, page_index: int, pdf_page_num: int,
                system_hint: str = "") -> dict:
    """Call GPT-4o vision with retry logic. Returns parsed JSON dict."""
    page_hint  = f"[PDF page {pdf_page_num}]\n\n"
    system_msg = _SYSTEM_PROMPT + ("\n\n" + system_hint if system_hint else "")
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=8192,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
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
                            {"type": "text", "text": page_hint + _PAGE_PROMPT},
                        ],
                    },
                ],
            )
            raw = response.choices[0].message.content or "{}"
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                try:
                    result = _repair_truncated_json(raw)
                    print(f"  [gpt4o] p{page_index+1}: JSON repaired (trailing comma or truncation)")
                    return result
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
    # Negative lookbehind: skip codes already inside [GELAKT: ...] to prevent
    # GPT-4o's own [GELAKT: 5.1.2e] from becoming [GELAKT: [GELAKT: 5.1.2e]].
    # "[GELAKT: " is exactly 9 chars — a fixed-length lookbehind.
    return re.sub(
        r"(?<!\[GELAKT: )5\.[12]\.[1-9][a-z]{0,2}",
        lambda m: f"[GELAKT: {m.group()}]",
        text,
        flags=re.IGNORECASE,
    )


def _normalise_doc_code(raw) -> str | None:
    """Validate and normalise a doc code string. Returns '0143' style or None."""
    if not raw:
        return None
    # Reject WOO redaction codes (5.1.x / 5.2.x) before stripping non-digits —
    # otherwise "5.1.2" → "512" → "0512" which creates false document boundaries.
    if re.match(r"5\.[12]\.", str(raw)):
        return None
    s = re.sub(r"\D", "", str(raw))   # digits only
    original_len = len(s)
    if len(s) == 7:
        # 7-digit barcode: extract digits 4-6 and prefix with 0
        s = "0" + s[3:6]
    elif len(s) > 4:
        # Multi-digit stamp (5, 6, 8-digit…): look for an embedded 0NNN code first
        m = re.search(r"0\d{3}", s)
        s = m.group() if m else s[-4:].zfill(4)
    elif len(s) >= 4:
        s = s[-4:].zfill(4)
    else:
        # 2–3 digit number — accept as a short WOO code (1–999) and zero-pad.
        # Handles "Doc 23" → GPT-4o returns "0023" per prompt, but also catches
        # cases where the model returns the raw digits only.
        n = int(s) if s.isdigit() else 0
        if 1 <= n <= 999:
            s = s.zfill(4)
        else:
            return None
    # Reject year-like numbers (1900–2099)
    if re.fullmatch(r"(19|20)\d{2}", s):
        return None
    # Reject 0000 — not a valid WOO document code (valid range 0001–9999)
    if s == "0000":
        return None
    # Reject codes that are the normalised form of WOO redaction articles:
    #   5.1.X → digits 51X → zero-padded → 051X  (X = 1–9)
    #   5.2.X → digits 52X → zero-padded → 052X
    # Covers both the case where GPT-4o returns the raw "5.1.2" string (caught above)
    # and the case where it returns the already-digit-stripped "0512" directly.
    if re.fullmatch(r"05[12][1-9]", s):
        return None
    # For 4-digit input: require WOO-style 0NNN (leading zero) to reject bare page counters
    if original_len == 4 and not re.fullmatch(r"0\d{3}", s):
        return None
    # For multi-digit input: accept any non-year 4-digit extraction
    if re.fullmatch(r"\d{4}", s):
        return s
    return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def save_cache(page_data: list[dict], cache_path: Path,
               dossier_profile: dict | None = None) -> None:
    """Save per-page text/metadata to JSON. Images are excluded (re-rendered on load)."""
    payload = [
        {
            "page_num":        p["page_num"],
            "text":            p["text"],
            "doc_code":        p["doc_code"],
            "is_new_document": p["is_new_document"],
            "within_doc_page": p["within_doc_page"],
            "category":        p["category"],
            "doc_subtype":     p.get("doc_subtype") or "other",
            "chat_name":       p.get("chat_name"),
            "chat_messages":   p.get("chat_messages") or [],
            "email_start":     p.get("email_start", False),
            "email_from":      p.get("email_from"),
            "email_to":        p.get("email_to"),
            "email_cc":        p.get("email_cc"),
            "email_subject":   p.get("email_subject"),
            "email_date":      p.get("email_date"),
            "doc_date":        p.get("doc_date"),
            "doc_sender":      p.get("doc_sender"),
        }
        for p in page_data
    ]
    root: dict = {"dpi": _RENDER_DPI, "pages": payload}
    if dossier_profile:
        root["dossier_profile"] = dossier_profile
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)
    print(f"[gpt4o] Progress saved to cache ({cache_path.name}).")


def _load_cache_pages(cache_path: Path) -> tuple[list[dict], int, list[dict], str | None]:
    """Load page metadata from JSON cache. Returns (pages_without_images, dpi, boundary_docs, stamp_format)."""
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    dpi          = data.get("dpi", _RENDER_DPI)
    pages        = data["pages"]
    boundary     = data.get("boundary_documents") or []
    stamp_format = (data.get("dossier_profile") or {}).get("stamp_format")
    # Re-normalise category in case cache was written before the fix
    for p in pages:
        p["category"] = _normalise_category(str(p.get("category") or "Other"))
        p["doc_subtype"] = _normalise_doc_subtype(p.get("doc_subtype"), p["category"])
    return pages, dpi, boundary, stamp_format


def _update_cache_boundaries(documents: list[dict], cache_path: Path) -> None:
    """Append pass-2 boundary decisions to an existing cache file."""
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        data["boundary_documents"] = documents
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[gpt4o] Document boundaries saved to cache.")
    except Exception as e:
        print(f"[gpt4o] Warning: could not update cache boundaries: {e}")


def _update_cache_emails(emails_by_doc: dict, cache_path: Path) -> None:
    """Store pass-3 full-document email extraction results in the cache file."""
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        data["emails_by_doc"] = emails_by_doc
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[gpt4o] Email extraction results saved to cache.")
    except Exception as e:
        print(f"[gpt4o] Warning: could not update cache emails: {e}")


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
    pages_meta, dpi, boundary_docs, stamp_format = _load_cache_pages(cache_path)
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

    return _finalise_pipeline(page_data, boundary_docs=boundary_docs or None,
                              stamp_format=stamp_format)


# ── OCR stamp fallback ───────────────────────────────────────────────────────

def _ocr_stamp_fallback(page_data: list[dict]) -> None:
    """
    For pages where GPT-4o returned no stamp code, attempt recovery using
    targeted tesseract OCR on the same corner regions as pipeline_ocr.

    Only runs when the dossier appears to use stamps (at least one page already
    has a detected code). Mutates page_data in-place.
    """
    stamped = [p for p in page_data if p.get("doc_code")]
    if not stamped:
        return  # stampless dossier — nothing to recover
    # Require at least 2 distinct codes before spreading via OCR fallback.
    # A single repeated code (e.g. all "0000") is a false-positive signal —
    # running OCR would only propagate the bad code to remaining pages.
    distinct_codes = {p["doc_code"] for p in stamped}
    if len(distinct_codes) < 2:
        return

    null_pages = [p for p in page_data if not p.get("doc_code") and p.get("image") is not None]
    if not null_pages:
        return

    from pipeline_ocr import _find_doc_code_raster

    recovered = 0
    for p in null_pages:
        code = _find_doc_code_raster(p["image"])
        if code:
            p["doc_code"] = code
            recovered += 1

    if recovered:
        print(f"[gpt4o] Recovered {recovered}/{len(null_pages)} missing stamp codes via OCR.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_pdf_vlm(
    pdf_path:   Path,
    api_key:    str | None = None,
    max_pages:  int | None = None,
    cache_path: Path | None = None,
    page_range: tuple[int, int] | None = None,
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
    print(f"[gpt4o] Converting PDF pages to images...")
    from pdf2image import convert_from_path
    images: list[Image.Image] = convert_from_path(str(pdf_path), dpi=_RENDER_DPI)

    all_images = images
    if page_range is not None:
        pr_start = max(1, page_range[0])
        pr_end   = min(len(all_images), page_range[1])
        images = all_images[pr_start - 1:pr_end]
        page_offset = pr_start - 1  # offset to get real PDF page numbers
        print(f"[gpt4o] Page range: {pr_start}–{pr_end} of {len(all_images)} total pages.")
    elif max_pages is not None:
        images = all_images[:max_pages]
        page_offset = 0
        print(f"[gpt4o] Test mode — processing first {len(images)} pages.")
    else:
        page_offset = 0

    total = len(images)
    print(f"[gpt4o] Step 2/4 — Reading all {total} pages with GPT-4o vision ({_MAX_WORKERS} pages at a time)...")

    # ── Raster stamp survey (before GPT-4o, free) ───────────────────────────
    print(f"[gpt4o] Step 1/4 — Pre-scanning {total} pages for document stamps (quick, no API cost)...")
    raster_survey = _raster_stamp_survey(images, page_offset)
    if raster_survey["has_stamps"]:
        lo, hi = raster_survey["range"]
        print(f"[gpt4o] Stamps found on {raster_survey['coverage']:.0%} of pages "
              f"— {raster_survey['distinct_count']} document codes ({lo}–{hi}).")
    else:
        print(f"[gpt4o] No consistent stamps found "
              f"({raster_survey['coverage']:.0%} of pages had a candidate, but below threshold).")

    # ── Pass 0: pilot run to profile stamp format ────────────────────────────
    # Only skip the GPT-4o pilot when raster coverage is near-zero (<= 5%).
    # Low-but-nonzero coverage (e.g. 17%) can indicate a DocNr-format dossier
    # where stamps appear only on the first page of each document — the pilot
    # must still run so it can detect the "docnr" format and inject the right hint.
    raster_confident_no_stamps = (
        not raster_survey["has_stamps"]
        and raster_survey["coverage"] <= 0.05
        and not raster_survey.get("has_docnr_labels")
    )
    if raster_confident_no_stamps:
        profile       = {"stamp_format": "none", "notes": "raster survey confirmed no stamps"}
        pilot_results = {}
        print("[gpt4o] Stamp sampling skipped — pre-scan confirmed this dossier has no stamps.")
    else:
        profile, pilot_results = _profile_dossier(images, client)

    profile_hint = _profile_to_hint(profile)
    raster_hint  = _raster_survey_to_hint(raster_survey, page_offset)
    system_hint  = "\n\n".join(h for h in [profile_hint, raster_hint] if h)

    # ── Stage 1: per-page GPT-4o analysis (parallel) ────────────────────────
    # Pilot pages already have results — only submit the remaining pages.
    def _process_page(args: tuple[int, Image.Image]) -> tuple[int, dict]:
        i, img = args
        image_b64 = _encode_image(img)
        return i, _call_gpt4o(image_b64, client, i, pdf_page_num=i + 1 + page_offset,
                               system_hint=system_hint)

    raw_results: dict[int, dict] = dict(pilot_results)  # seed with pilot data
    completed = len(pilot_results)

    remaining = [(i, img) for i, img in enumerate(images) if i not in pilot_results]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {executor.submit(_process_page, (i, img)): i for i, img in remaining}
        for future in as_completed(futures):
            i, result = future.result()
            completed += 1
            print(f"[gpt4o] Page {completed}/{total} read.")
            raw_results[i] = result

    page_data: list[dict] = []
    for i, img in enumerate(images):
        result = raw_results.get(i, {})

        text            = str(result.get("text") or "").strip()
        is_new_doc      = bool(result.get("is_new_document", False))
        doc_code        = _normalise_doc_code(result.get("doc_code"))
        within_doc_page = result.get("within_doc_page")
        category        = _normalise_category(str(result.get("category") or "Other"))
        doc_subtype     = _normalise_doc_subtype(result.get("doc_subtype"), category)

        if within_doc_page is not None:
            try:
                within_doc_page = int(within_doc_page)
            except (ValueError, TypeError):
                within_doc_page = None

        chat_name     = result.get("chat_name") or None
        chat_messages = result.get("chat_messages") or []
        if not isinstance(chat_messages, list):
            chat_messages = []

        email_start   = bool(result.get("email_start", False)) if category == "E-mail" else False
        email_from    = result.get("email_from")    or None
        email_to      = result.get("email_to")      or None
        email_cc      = result.get("email_cc")      or None
        email_subject = result.get("email_subject") or None
        email_date    = result.get("email_date")    or None
        doc_date      = result.get("doc_date")      or None
        doc_sender    = result.get("doc_sender")    or None

        page_data.append({
            "page_num":        i + 1 + page_offset,
            "image":           img,
            "text":            text,
            "doc_code":        doc_code,
            "is_new_document": is_new_doc,
            "within_doc_page": within_doc_page,
            "category":        category,
            "doc_subtype":     doc_subtype,
            "doc_date":        doc_date,
            "doc_sender":      doc_sender,
            "chat_name":       chat_name,
            "chat_messages":   chat_messages,
            "email_start":     email_start,
            "email_from":      email_from,
            "email_to":        email_to,
            "email_cc":        email_cc,
            "email_subject":   email_subject,
            "email_date":      email_date,
        })

    # ── Attach raster codes to page_data ────────────────────────────────────
    raster_codes = raster_survey.get("codes", [])
    for p in page_data:
        idx = p["page_num"] - 1 - page_offset
        p["raster_code"] = raster_codes[idx] if 0 <= idx < len(raster_codes) else None

    # ── OCR stamp fallback ───────────────────────────────────────────────────
    # For pages where GPT-4o missed the stamp, run targeted corner-crop OCR.
    # Only runs when the dossier has stamps on at least some pages.
    _ocr_stamp_fallback(page_data)

    # ── Save cache ───────────────────────────────────────────────────────────
    if cache_path:
        save_cache(page_data, cache_path, dossier_profile=profile)

    return _finalise_pipeline(page_data, client=client, cache_path=cache_path,
                              stamp_format=profile.get("stamp_format"))


def _backfill_email_pdf_pages(emails: list[dict], email_pages: list[dict]) -> None:
    """
    Attach pdf_page to emails produced by _extract_emails_full_doc (pass-3),
    which sees only text and has no page info.

    Strategy: walk email_pages in order; each entry with email_start=True is
    the start of a new email. Match to the emails list positionally — pass-3
    returns emails in the same order as the thread appears in the PDF.
    """
    start_pages = [p for p in email_pages if p.get("email_start") or (email_pages.index(p) == 0 and not any(q.get("email_start") for q in email_pages))]
    # Simpler: collect pdf_page for every email_start page in order
    start_pdf_pages = [p.get("pdf_page") for p in email_pages if p.get("email_start")]
    if not start_pdf_pages:
        # No email_start markers — use first page of the doc for all
        first = next((p.get("pdf_page") for p in email_pages if p.get("pdf_page")), None)
        start_pdf_pages = [first] * len(emails)
    for i, email in enumerate(emails):
        if "pdf_page" not in email or email["pdf_page"] is None:
            email["pdf_page"] = start_pdf_pages[i] if i < len(start_pdf_pages) else start_pdf_pages[-1]


def _build_emails_from_pages(doc_code: str, email_pages: list[dict]) -> list[dict]:
    """Assemble individual email dicts from per-page GPT-4o email metadata."""
    emails: list[dict] = []
    current: dict | None = None
    email_idx = 0

    def _split_email_datetime(value: str | None) -> tuple[str | None, str | None]:
        if not value:
            return None, None
        if "T" in value:
            date_part, time_part = value.split("T", 1)
            return date_part, time_part[:5]
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})", value)
        if m:
            return m.group(1), m.group(2)
        return value, None

    for p in email_pages:
        if p.get("email_start") or current is None:
            if current is not None:
                emails.append(current)
            email_idx += 1
            date_part, time_part = _split_email_datetime(p.get("email_date"))
            current = {
                "id":       f"{doc_code}.{email_idx}",
                "subject":  p.get("email_subject"),
                "sender":   p.get("email_from"),
                "to":       p.get("email_to"),
                "cc":       p.get("email_cc"),
                "date":     date_part,
                "time":     time_part,
                "text":     p["text"],
                "pdf_page": p.get("pdf_page"),  # PDF page where this email starts
            }
        else:
            current["text"] += "\n\n" + p["text"]

    if current:
        emails.append(current)

    return emails


# ── Pass 2: context-aware boundary detection ──────────────────────────────────

_BOUNDARY_SYSTEM = (
    "You are analyzing a Dutch government WOO (Wet Open Overheid) disclosure dossier. "
    "You receive a compact per-page summary and must draw document boundaries and, for "
    "email documents, individual email boundaries. "
    "Not all dossiers use inventory stamps. When stamps are consistently present, a stamp code "
    "change between pages is a primary boundary signal on its own — it alone is sufficient to "
    "start a new document. When no stamps are present, rely on content signals instead. "
    "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
)

_BOUNDARY_PROMPT = """\
Below is a {n}-page WOO dossier summary. Each line:
  p<num>: stamp=<4-digit code|null> wpn=<within-doc page|null> new=<Y|N> cont=<Y|N> cat=<category> eml=<Y|N> | "<first 500 chars of text>"
  (new=Y means pass-1 suggested a new doc starts here; cont=Y means the page text begins mid-sentence — a near-certain continuation; eml=Y means email header fields were detected)

{summary}

Return ONLY this JSON (all page numbers are the p<num> indices above, 1-based):
{{
  "documents": [
    {{
      "doc_code": "<stamp code if consistently present for this document, else null>",
      "start_page": <integer>,
      "email_starts": [<page numbers where individual emails start — include the first page of email docs>]
    }}
  ]
}}

DOCUMENT BOUNDARY RULES — apply signals in priority order:

1. DEFINITIVE (never override):
   - cont=Y means a sentence continued from the previous page. This page CANNOT start a new document.
   - Every page must belong to exactly one document; list documents in ascending start_page order with no gaps.
   - A page that immediately follows an Inventarislijst segment ALWAYS starts a new document,
     even if cont=Y, even if no stamp is present. The Inventarislijst is always document 1;
     everything after it is a separate document.

2. STAMP CODES — when stamps are present they are the PRIMARY split signal:
   - Stamp code CHANGES between consecutive pages: this alone is sufficient to start a new document.
     Exception: cont=Y (rule 1) overrides even a stamp change.
   - Same stamp on consecutive pages: these pages belong to the same document (do NOT split),
     even if content signals suggest otherwise.
   - Incrementing sequential numbers that change by 1 on every page (e.g. 00001→00002→00003)
     are per-page sequence counters, NOT document codes — ignore them for boundary decisions.
   - stamp=null throughout (no stamps in this dossier): skip to rules 3–5 entirely.
   - A stamp=null page following stamped pages with no content split signals: treat as continuation.

3. STRONGEST content signals (each alone is sufficient when no stamps are present):
   - wpn=1 AND new=Y together confirm a new document start.
   - The text preview begins with "Pagina 1 van" or "1/" in a footer/stamp context.
   - The text preview shows a complete, fresh document header at the very top: a full
     Van/Aan/Onderwerp email block, a new memo heading with date + reference, or a new
     letter/rapport title page.
   - A clear category transition between consecutive pages (e.g. Inventarislijst → E-mail,
     E-mail → Nota, Other → E-mail) when the new page's text preview shows a fresh document
     opening (email header, new letterhead, new title). Category alone — without a content
     signal — is not sufficient; but category + any visible document start in the text is.

4. MODERATE content signals (combine two or more to split when no stamps are present):
   - new=Y alone (pass-1 detected a new header, but could be a quoted/forwarded header in a body)
   - wpn=1 alone (within-doc page resets to 1)
   - Category changes significantly between consecutive pages without a visible new document start
   - Text preview clearly shows a new letterhead, new ministry logo, or new document reference

5. DEFAULT — when in doubt, keep pages together. Spurious splits create empty documents;
   missed splits only merge two documents. Merging is the safer error.

EMAIL BOUNDARY RULES:
- email_starts: list page numbers where a genuinely NEW email begins — a fresh
  Van:/From: + Aan:/To: + Onderwerp:/Subject: header block at the top of the page.
- eml=Y fires on QUOTED and FORWARDED headers too — do NOT treat eml=Y alone as a new email start.
- A continuation page (quoted reply chain, body text, page 2+ of a long email) is NOT a new email
  start even if eml=Y.
- One email can span multiple pages; do not create multiple email_starts for the same email.
- Always include the first page of an email-type document in email_starts.
- Leave email_starts empty ([]) for non-email documents.
"""


# ── Full-document email extraction (pass 3) ───────────────────────────────────

_EMAIL_EXTRACT_SYSTEM = (
    "You are a structured data extractor for Dutch government WOO (Wet Open Overheid) email documents. "
    "Extract every individual email from the provided document text with accurate metadata. "
    "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
)

_EMAIL_EXTRACT_PROMPT = """\
Below is the full text of WOO document {code} — an email document (possibly a thread).

Extract every individual email and return:
{{
  "emails": [
    {{
      "subject":     "<Onderwerp/Subject/Betreft line, or null>",
      "sender":      "<From/Van/Afzender — name only, no email address; '[GELAKT]' if redacted>",
      "to":          "<To/Aan field, or null>",
      "cc":          "<CC field, or null>",
            "date":        "<date as YYYY-MM-DD — parse any format; null if not found>",
            "time":        "<time as HH:MM if visible in the source, else null>",
      "attachments": ["<filename or description>"],
      "body":        "<email body text only — no header lines>"
    }}
  ]
}}

Extraction rules:
- List emails oldest-first (in a reply chain the quoted original is oldest).
- A new email starts at a Van:/From:/Afzender: + Aan:/To: + Onderwerp:/Subject:/Betreft: header block,
  OR at a separator line like "-----Oorspronkelijk bericht-----" / "-----Original Message-----".
- A reply body appearing BEFORE the first quoted header is the newest email — list it LAST.
- date: parse ALL Dutch and European formats to YYYY-MM-DD:
    "3 januari 2024"       → "2024-01-03"
    "maandag 6 mei 2024"   → "2024-05-06"
    "06-03-2024"           → "2024-03-06"
    "DD/MM/YYYY"           → "YYYY-MM-DD"
  Return null only if truly no date is present.
- time: if a visible sent time exists, return it as HH:MM. Otherwise return null.
- attachments: collect filenames of files attached to this email. They appear in two ways:
    1. After a "Bijlage(n):" / "Attachment(s):" / "Bijlage:" label — list each filename
    2. As bare standalone lines (inside or immediately after the header block, or at the end of the
       email body) that consist of nothing but a filename with a document extension, e.g.:
         "Rapport_Q3.pdf"
         "overzicht kiosken.xlsx"
         "brief aan gemeente.docx"
       Recognised extensions: .pdf .doc .docx .xls .xlsx .ppt .pptx .msg .eml .zip .csv .txt .png .jpg .jpeg
  Do NOT include sentences from the body that merely mention a filename.
  Return [] if no attachments are found.
- body: message body only — strip all Van/Aan/CC/Onderwerp/Datum header lines from the top.
  Preserve [GELAKT: 5.1.2e] markers exactly as found.
- sender: name only. Strip angle brackets and email addresses entirely.
  If redacted, use "[GELAKT]".
- If this is a single email (no thread), return an array with one item.

Document text:
{text}
"""

_EMAIL_EXTRACT_MAX_CHARS = 24_000   # ~6 000 tokens — keeps cost low for gpt-4o-mini


def _extract_emails_full_doc(code: str, text: str, client) -> list[dict]:
    """
    Full-document email extraction via GPT-4o-mini.

    Sends the complete email document text in a single call so GPT can see
    the whole thread, correctly identify split points, parse Dutch dates,
    and extract attachment lists — without page-boundary fragmentation.

    Falls back to an empty list on failure; the caller should then fall back
    to _build_emails_from_pages().
    """
    truncated = text[:_EMAIL_EXTRACT_MAX_CHARS]
    prompt    = _EMAIL_EXTRACT_PROMPT.format(code=code, text=truncated)

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_BOUNDARY_MODEL,   # gpt-4o-mini — text only, no vision needed
                max_tokens=16384,        # gpt-4o-mini max — large threads can produce long JSON
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _EMAIL_EXTRACT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                timeout=90.0,            # prevent silent hang on slow/long responses
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                # Response was truncated — JSON will be invalid; no point retrying
                # with the same input. Fall back to per-page assembly.
                print(f"  [gpt4o-email] Doc {code}: email thread too long for single pass — using page-by-page fallback.")
                return []
            data = json.loads(choice.message.content or "{}")
            raw  = data.get("emails") or []
            result: list[dict] = []
            for i, em in enumerate(raw, 1):
                result.append({
                    "id":          f"{code}.{i}",
                    "subject":     em.get("subject")     or None,
                    "sender":      em.get("sender")      or None,
                    "to":          em.get("to")          or None,
                    "cc":          em.get("cc")          or None,
                    "date":        em.get("date")        or None,
                    "time":        em.get("time")        or None,
                    "attachments": em.get("attachments") or [],
                    "text":        em.get("body")        or "",
                })
            print(f"  [gpt4o-email] Doc {code}: found {len(result)} email(s).")
            return result
        except json.JSONDecodeError as e:
            print(f"  [gpt4o-email] Doc {code}: unexpected response format (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"  [gpt4o-email] Doc {code}: API error (attempt {attempt + 1}): {str(e)[:120]}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return []


# ── Inventarislijst boundary hint ─────────────────────────────────────────────

# Header lines to skip when parsing inventarislijst rows
_INV_SKIP_RE = re.compile(
    r'^\s*(?:ProcesId|Document\s+nummer|Bestandsnaam|Inventarislijst|'
    r'Titel|Nr\.?\s|Nummer|Kenmerk|Datum|Pagina|Omschrijving|'
    r'Besluit|Weigeringsgrond|Openbaarheid|#\s)',
    re.IGNORECASE,
)
# Data row: optional procesId (3–7 digits) + doc number (1–4 digits) + title
_INV_ROW_RE = re.compile(
    r'^\s*(?:\d{3,7}\s+)?'   # optional procesId (e.g. 6839)
    r'(0?\d{1,4})\b\s+'      # doc number
    r'(.{4,})',               # rest of line (title + optional grounds)
)
# Trailing transparency decisions / WOO grounds to strip from the end of the title.
# \bBR\b prevents matching word-starts like "brief" or "Brf" — BR is a standalone code.
_INV_GROUNDS_RE = re.compile(
    r'\s+(?:5\.[12]\.\d\w*|\bBR\b|niet\s+openbaar|gedeeltelijk\s+openbaar|openbaar).*$',
    re.IGNORECASE,
)


def _extract_inventarislijst_hint(page_data: list[dict]) -> str:
    """
    Scan pass-1 output for Inventarislijst pages and extract the document list.
    Detection uses the category field set by the VLM — no header keyword required,
    so tables without an 'Inventarislijst' heading are handled correctly.
    Returns a formatted hint string for the pass-2 boundary prompt, or "" if none found.
    """
    inv_pages = [
        p for p in page_data
        if p.get("category") == "Inventarislijst" and p.get("text", "").strip()
    ]
    if not inv_pages:
        return ""

    items: list[tuple[str, str]] = []
    seen:  set[str] = set()

    for p in inv_pages:
        for line in p.get("text", "").splitlines():
            line = line.strip()
            if not line or _INV_SKIP_RE.match(line):
                continue
            m = _INV_ROW_RE.match(line)
            if not m:
                continue
            num_raw = m.group(1)
            title   = _INV_GROUNDS_RE.sub("", m.group(2)).strip().rstrip(".,")
            if not num_raw.isdigit() or len(title) < 4:
                continue
            num = num_raw.zfill(4)
            if num in seen:
                continue
            seen.add(num)
            items.append((num, title))

    if not items:
        return ""

    out = [
        f"INVENTARISLIJST — this dossier contains {len(items)} document(s).",
        "When a page's email subject, letter heading, or document title matches one of",
        "the entries below, treat it as a boundary signal for that document:",
        "",
    ]
    for num, title in sorted(items):
        out.append(f"  [{num}] {title}")
    return "\n".join(out)


_STAMP_NOISE_RE = re.compile(r"^\s*(?:\d{4,}|[A-Z]{2,}\d+|\[PDF page \d+\])\s*", re.IGNORECASE)


def _continuation_flag(text: str) -> str:
    """Return 'Y' if the page text starts mid-sentence (strong continuation signal)."""
    # Strip leading stamp/barcode noise to get to the actual first word
    clean = _STAMP_NOISE_RE.sub("", text).lstrip()
    if not clean:
        return "N"
    # Starts with a lowercase letter → almost certainly continues from previous page
    return "Y" if clean[0].islower() else "N"


def _build_page_summary(page_data: list[dict]) -> str:
    """Build the compact per-page text table sent to pass 2."""
    lines = []
    for i, p in enumerate(page_data, 1):
        stamp   = p["doc_code"] or "null"
        wpn     = str(p["within_doc_page"]) if p["within_doc_page"] is not None else "null"
        new     = "Y" if p.get("is_new_document") else "N"
        cat     = p.get("category") or "Other"
        eml     = "Y" if p.get("email_start") else "N"
        text    = p.get("text") or ""
        cont    = _continuation_flag(text)
        extra   = ""
        if p.get("email_start"):
            if p.get("email_from"):
                extra += f' from="{str(p["email_from"])[:30]}"'
            if p.get("email_subject"):
                extra += f' subj="{str(p["email_subject"])[:40]}"'
        preview = " ".join(text.split())[:500]
        lines.append(
            f'p{i:03d}: stamp={stamp} wpn={wpn} new={new} cont={cont} cat={cat} eml={eml}{extra} | "{preview}"'
        )
    return "\n".join(lines)


def _call_boundary_pass(summary: str, n_pages: int, client, inv_hint: str = "") -> list[dict]:
    """Single GPT-4o-mini text-only call that returns document + email boundaries."""
    hint_prefix = inv_hint + "\n\n" if inv_hint else ""
    prompt = hint_prefix + _BOUNDARY_PROMPT.format(n=n_pages, summary=summary)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_BOUNDARY_MODEL,
                max_tokens=4096,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _BOUNDARY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return data.get("documents") or []
        except json.JSONDecodeError as e:
            print(f"  [gpt4o-pass2] JSON parse error (attempt {attempt + 1}): {e}")
        except Exception as e:
            print(f"  [gpt4o-pass2] API error (attempt {attempt + 1}): {str(e)[:120]}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return []


def _build_docs_from_boundaries(
    page_data: list[dict],
    documents: list[dict],
) -> dict[str, dict]:
    """Assign pages to documents based on pass-2 boundary decisions."""
    if not documents:
        docs_raw: dict[str, dict] = {}
        for p in page_data:
            _append_page(docs_raw, "auto_001", p)
        return docs_raw

    n           = len(page_data)
    docs_sorted = sorted(documents, key=lambda d: int(d.get("start_page", 1)))

    idx_to_code: dict[int, str] = {}
    email_start_idxs: set[int]  = set()

    for i, doc in enumerate(docs_sorted):
        start    = int(doc.get("start_page", 1))
        end      = (int(docs_sorted[i + 1].get("start_page", n + 1)) - 1
                    if i + 1 < len(docs_sorted) else n)
        raw_code = doc.get("doc_code")
        code     = (_normalise_doc_code(raw_code) if raw_code else None) or f"auto_{i + 1:03d}"
        for idx in range(start, end + 1):
            idx_to_code[idx] = code
        for ep in (doc.get("email_starts") or []):
            email_start_idxs.add(int(ep))

    docs_raw: dict[str, dict] = {}
    for seq_idx, p in enumerate(page_data, 1):
        code  = idx_to_code.get(seq_idx, f"auto_{seq_idx:03d}")
        p_mod = dict(p)
        # Override pass-1 email_start with pass-2 decisions (more context-aware)
        p_mod["email_start"] = seq_idx in email_start_idxs
        _append_page(docs_raw, code, p_mod)

    return docs_raw


def _stamp_coverage(page_data: list[dict]) -> float:
    """Fraction of pages that have a detected stamp code (GPT-4o or raster)."""
    if not page_data:
        return 0.0
    return sum(1 for p in page_data if p.get("doc_code") or p.get("raster_code")) / len(page_data)


def _finalise_pipeline(
    page_data: list[dict],
    client=None,
    cache_path: Path | None = None,
    boundary_docs: list[dict] | None = None,
    stamp_format: str | None = None,
) -> dict[str, dict]:
    """Stage 2 + 3: assign doc codes and build the final docs dict.

    Priority:
      1. Stamp forward-fill — when stamps cover >= threshold of pages.
         Reliable and cheap; no LLM needed.
      2. LLM pass-2 boundary detection — when stamps are absent/sparse.
         Uses cached boundary_docs if available, else calls the API.
      3. VLM is_new_document signal — when no client and no stamps.
    """
    # DocNr-format stamps appear on the first page of each document only, so
    # coverage is structurally bounded by 1/avg_doc_length — a lower threshold
    # is needed to avoid incorrectly routing these dossiers to pass-2.
    threshold = 0.15 if stamp_format == "docnr" else _STAMP_THRESHOLD
    coverage = _stamp_coverage(page_data)
    has_reliable_stamps = coverage >= threshold

    if has_reliable_stamps:
        print(f"[gpt4o] Step 3/4 — Assigning documents by stamp codes ({coverage:.0%} of pages stamped).")
        docs_raw = _build_docs_forward_fill(page_data)
        method   = "gpt4o-stamp"
    elif boundary_docs is not None:
        # Cached LLM boundary decisions from a previous pass-2 run
        print(f"[gpt4o] Step 3/4 — Assigning documents using saved boundary decisions (stamps sparse/absent).")
        docs_raw = _build_docs_from_boundaries(page_data, boundary_docs)
        method   = "gpt4o-2pass"
    elif client is not None:
        print(f"[gpt4o] Step 3/4 — Stamp coverage {coverage:.0%} below threshold ({threshold:.0%}) — asking AI to find document boundaries...")
        inv_hint  = _extract_inventarislijst_hint(page_data)
        if inv_hint:
            n_docs = sum(1 for l in inv_hint.splitlines() if l.startswith("  ["))
            print(f"[gpt4o] Found an inventory table listing {n_docs} document(s) — using it to guide boundary detection.")
        summary   = _build_page_summary(page_data)
        documents = _call_boundary_pass(summary, len(page_data), client, inv_hint=inv_hint)
        if documents:
            docs_raw = _build_docs_from_boundaries(page_data, documents)
            method   = "gpt4o-2pass"
            if cache_path:
                _update_cache_boundaries(documents, cache_path)
        else:
            print("[gpt4o] Boundary detection failed — falling back to per-page signals.")
            docs_raw = _build_docs_boundary(page_data)
            method   = "gpt4o-boundary"
    else:
        # No client, no cached boundaries, no stamps
        print("[gpt4o] No stamps and no API client — assigning documents using per-page signals only.")
        docs_raw = _build_docs_boundary(page_data)
        method   = "gpt4o-boundary"

    n_email_docs = sum(
        1 for d in docs_raw.values()
        if next((c for c in d["categories"] if c != "Other"), "Other") == "E-mail"
    )
    if n_email_docs and client is not None:
        print(f"[gpt4o] Step 4/4 — Extracting email threads from {n_email_docs} email document(s)...")
    elif not n_email_docs:
        print(f"[gpt4o] Step 4/4 — No email documents found, skipping email extraction.")

    # ── Phase A: pre-compute all non-API metadata (fast, no I/O) ────────────────
    doc_meta: dict[str, dict] = {}
    for code, d in docs_raw.items():
        full_text       = "\n\n".join(d["texts"])
        annotated       = _annotate_text(full_text)
        redaction_codes = _count_redaction_codes(full_text)

        # First non-"Other" page category wins: email headers only appear on the
        # first page, so majority vote would wrongly demote multi-page emails.
        category    = next((c for c in d["categories"] if c != "Other"), "Other")
        doc_subtype = next((s for s in d.get("doc_subtypes", []) if s and s != "other"), None)
        if not doc_subtype:
            doc_subtype = _CATEGORY_TO_SUBTYPE.get(category, "other")

        chat_names    = d.get("chat_names") or []
        chat_name     = Counter(chat_names).most_common(1)[0][0] if chat_names else None
        doc_date      = next((v for v in d.get("doc_dates",   []) if v), None)
        doc_sender    = next((v for v in d.get("doc_senders", []) if v), None)

        doc_meta[code] = {
            "d":              d,
            "full_text":      full_text,
            "annotated":      annotated,
            "redaction_codes": redaction_codes,
            "category":       category,
            "doc_subtype":    doc_subtype,
            "chat_name":      chat_name,
            "chat_messages":  d.get("chat_messages") or [],
            "doc_date":       doc_date,
            "doc_sender":     doc_sender,
        }

    # ── Phase B: extract emails concurrently for all E-mail docs ─────────────
    email_results: dict[str, list] = {}
    email_codes = [
        code for code, m in doc_meta.items()
        if m["category"] == "E-mail" and client is not None
    ]

    def _extract_email_worker(code: str) -> tuple[str, list]:
        print(f"  [gpt4o-email] Doc {code}: extracting email thread...")
        return code, _extract_emails_full_doc(code, doc_meta[code]["annotated"], client)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {executor.submit(_extract_email_worker, code): code for code in email_codes}
        for future in as_completed(futures):
            code, result = future.result()
            email_results[code] = result

    # ── Phase C: assemble final docs dict ────────────────────────────────────
    docs: dict[str, dict] = {}
    for code, m in doc_meta.items():
        d        = m["d"]
        category = m["category"]

        # Build structured emails: prefer concurrent GPT-4o-mini result,
        # fall back to per-page assembly when extraction failed or no client.
        if category == "E-mail":
            if client is not None:
                structured_emails = email_results.get(code) or \
                    _build_emails_from_pages(code, d.get("email_pages", []))
                # Pass-3 emails have no pdf_page — backfill from email_pages
                if structured_emails and email_results.get(code):
                    _backfill_email_pdf_pages(structured_emails, d.get("email_pages", []))
            else:
                structured_emails = _build_emails_from_pages(code, d.get("email_pages", []))
        else:
            structured_emails = []

        docs[code] = {
            "doc_code":         code,
            "pages":            d["pages"],
            "pdf_pages":        d.get("pdf_pages") or [],
            "page_nums_in_doc": d["page_nums_in_doc"],
            "text":             m["full_text"],
            "annotated_text":   m["annotated"],
            "redaction_codes":  m["redaction_codes"],
            "category":         category,
            "doc_subtype":      m["doc_subtype"],
            "method":           method,
            "doc_date":         m["doc_date"],
            "doc_sender":       m["doc_sender"],
            "chat_name":        m["chat_name"],
            "chat_messages":    m["chat_messages"],
            "emails":           structured_emails,
        }

    if cache_path:
        emails_by_doc = {
            code: doc["emails"]
            for code, doc in docs.items()
            if doc.get("emails")
        }
        if emails_by_doc:
            _update_cache_emails(emails_by_doc, cache_path)

    total = len(page_data)
    print(f"[gpt4o] Done! Found {len(docs)} documents across {total} pages.")
    return docs


# ── Assignment helpers ────────────────────────────────────────────────────────

def _build_docs_forward_fill(page_data: list[dict]) -> dict[str, dict]:
    """Stamp-based assignment with forward-fill. Same logic as pipeline_ocr.

    Extra rule: when multiple nota/letter-like documents share the same stamp code,
    split them on is_new_document boundaries (category must be in _NOTA_LIKE to
    avoid false splits in email threads or chat exports).
    """
    docs_raw: dict[str, dict] = {}
    last_detected = None   # raw stamp from previous page (for change detection)
    last_code     = None   # effective key in use (may carry a _N suffix for sub-splits)
    unknown_count = 0
    sub_counts: dict[str, int] = {}  # base_code → number of sub-splits created so far

    for p in page_data:
        # Prefer GPT-4o doc_code; fall back to raster_code when GPT-4o missed the stamp
        detected = p["doc_code"] or p.get("raster_code")
        wpn      = p["within_doc_page"]
        category = p.get("category", "Other")
        # Never start a new doc if VLM explicitly says this is page 2+
        is_new   = (p["is_new_document"] or wpn == 1) and (wpn is None or wpn == 1)

        if detected and detected != last_detected:
            # Stamp changed — unambiguous new document
            current_code  = detected
            last_detected = detected
            last_code     = detected
        elif detected and is_new and category in _NOTA_LIKE:
            # Same stamp but VLM sees a new nota/letter heading — create a sub-split.
            # Only done for nota-like categories; email/chat threads share a code by design.
            n = sub_counts.get(detected, 1) + 1
            sub_counts[detected] = n
            current_code  = f"{detected}_{n}"
            last_detected = detected   # still tracking same base stamp
            last_code     = current_code
        elif detected:
            # Same stamp, continuation — keep using the current effective key
            current_code  = last_code or detected
            last_detected = detected
        elif is_new and last_code is not None:
            # Boundary but no stamp — create unknown slot
            unknown_count += 1
            current_code  = f"unknown_{unknown_count}"
            last_detected = None
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
            "pdf_pages":        [],
            "page_nums_in_doc": [],
            "texts":            [],
            "categories":       [],
            "doc_subtypes":     [],
            "doc_dates":        [],   # per-page doc_date values; first non-null wins
            "doc_senders":      [],   # per-page doc_sender values; first non-null wins
            "chat_names":       [],
            "chat_messages":    [],
            "email_pages":      [],
        }
    docs_raw[code]["pages"].append(p["image"])
    docs_raw[code]["pdf_pages"].append(p["page_num"])
    docs_raw[code]["page_nums_in_doc"].append(p["within_doc_page"])
    docs_raw[code]["texts"].append(p["text"])
    docs_raw[code]["categories"].append(p["category"])
    docs_raw[code]["doc_subtypes"].append(_normalise_doc_subtype(p.get("doc_subtype"), p.get("category", "Other")))
    if p.get("doc_date"):
        docs_raw[code]["doc_dates"].append(p["doc_date"])
    if p.get("doc_sender"):
        docs_raw[code]["doc_senders"].append(p["doc_sender"])
    if p.get("chat_name"):
        docs_raw[code]["chat_names"].append(p["chat_name"])
    docs_raw[code]["chat_messages"].extend(p.get("chat_messages") or [])
    docs_raw[code]["email_pages"].append({
        "email_start":   p.get("email_start", False),
        "email_from":    p.get("email_from"),
        "email_to":      p.get("email_to"),
        "email_cc":      p.get("email_cc"),
        "email_subject": p.get("email_subject"),
        "email_date":    p.get("email_date"),
        "text":          p["text"],
        "pdf_page":      p.get("page_num"),   # absolute PDF page number for jump-to
    })
