# WOO Organizer

Converts Dutch WOO (*Wet Open Overheid*) disclosure PDFs into an interactive HTML timeline. Documents are split, dated, categorised, and rendered with inline email cards, page thumbnails, recipient tags, attachment markers, and redaction-code summaries.

---

## Pipelines

Two independent pipelines share the same date-extraction, sorting, and visualisation layer. They differ in how they extract text and detect document boundaries.

| Pipeline | Entry point | How it works | Requires |
|---|---|---|---|
| OCR | `main.py` | pdfplumber text extraction + Tesseract OCR; rule-based boundary detection | Tesseract, Poppler |
| GPT-4o | `main_gpt4o.py` | GPT-4o vision per page (pass 1) → GPT-4o-mini boundary pass (pass 2) → GPT-4o-mini email extraction (pass 3) | OpenAI API key |

Both produce the same output schema and are directly benchmarkable against each other using `annotate.py`.

**When to use which:**
- OCR is free, fast on cached/searchable PDFs, and reaches near-GPT-4o accuracy on all-email dossiers with consistent stamps.
- GPT-4o handles mixed document types (Nota, Brief, Report alongside emails) better because it sees the visual layout of each page, not just extracted text. Expect a wider gap on diverse dossiers.

---

## Setup

**System dependencies (macOS)**
```bash
brew install tesseract poppler
```

**Python environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**API key** (GPT-4o pipeline only)
```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage

### OCR pipeline
```bash
python main.py                                      # test.pdf → woo_timeline.html
python main.py --pdf my_dossier.pdf --out out.html
```

### GPT-4o pipeline
```bash
# Full run — processes every page, saves a cache automatically
python main_gpt4o.py --pdf my_dossier.pdf

# Test run — first 15 pages only (saves to a separate cache)
python main_gpt4o.py --pdf my_dossier.pdf --vlm-pages 15

# Regenerate HTML from an existing cache (no API cost)
python main_gpt4o.py --pdf my_dossier.pdf --from-cache my_dossier_gpt4o_cache.json
```

### WOOLens server (browser UI)
```bash
python server.py          # serves WOOLens at http://localhost:8000
```
Upload a PDF in the browser and choose OCR or GPT-4o. Results stream in via SSE and render as an interactive timeline.

---

## Pipeline walkthrough

Both pipelines run the same three steps in sequence. Only Step 1 differs.

---

### Step 1 — Text extraction and document splitting

---

#### OCR pipeline (`pipeline_ocr.py`)

**1a. Searchability check**

`_is_searchable()` determines whether any page has a selectable text layer. Searchable PDFs use pdfplumber's direct extraction (fast); image-based PDFs fall back to full Tesseract OCR at 300 DPI.

**1b. Stage 1 — Per-page data collection**

For every page the pipeline collects:

**Doc code** — the 4-digit WOO stamp (e.g. `0144`) that identifies which document a page belongs to.

- *Text layer:* `_find_doc_code_words()` scans the top strip using pdfplumber's word list (regions: top-right → top-left → top-full). Year numbers (1900–2099) are excluded.
- *Raster OCR:* `_find_doc_code_raster()` renders the page at 200 DPI and OCRs six corner/edge regions in priority order (top-right → top-left → bottom-right → bottom-middle → bottom-left → top-full) with a digits-only Tesseract whitelist. Two-stage matching:
  1. Primary: `\b(0\d{3})\b` — requires a leading zero, rejecting article-number fragments.
  2. Fallback (bottom regions only): `\d{2,}(0\d{3})` — handles barcode strings like `7601430` → `0143`.
- Raster is always tried at `within_doc_page == 1` boundaries or when the text layer found nothing.

**Within-doc page number** — `_find_within_doc_page_raster()` OCRs the bottom-right corner to find the per-document page counter.

**Page text** — `page.extract_text()` for searchable PDFs; Tesseract (Dutch+English) for image-based pages or pages with fewer than 20 text characters. With `ocr_supplement=True`, an additional 300 DPI OCR pass supplements the text-layer redaction code count.

**1c. Stage 2 — Document code assignment**

- **Stamps detected** → `_build_docs_forward_fill()`: pages inherit the most recently seen stamp code. Pages where `_is_new_doc_boundary()` fires (`within_doc_page == 1` or `"Pagina 1 van"` in text) but no code is readable get an auto-generated `unknown_N` code.
- **No stamps anywhere** → auto-split mode with three heuristic signals (in priority order):
  1. `within_doc_page == 1` or `"Pagina 1 van"` in text
  2. `_is_fresh_email_start()` — detects a new email document
  3. `_is_fresh_nota_start()` — detects a new nota/brief

**Email boundary detection** (`_is_fresh_email_start`):

1. *First-line veto:* if the first non-empty line is neither an email header field nor a document stamp, the page is a continuation — prevents false positives from quoted reply chains.
2. Count ≥ 2 header fields (`Van`/`From`/`Aan`/`To`/`Onderwerp`/`Subject`/`Datum`/`Date`/`Sent`/`Verzonden`/`Betreft`/`Afzender`/`CC`) in the first 20 lines.
3. Stamp matching is OCR-noise-tolerant: `D→O`, `r→,`, and `5↔S` substitutions are handled (e.g. `Oocnr28`, `Oocn,59`, `DocnrSS` all match correctly).

**Nota/brief boundary detection** (`_is_fresh_nota_start`):

Same first-line veto + ≥ 2 of: `Contactpersoon`, `Datum`, `Onze referentie`, `Ons kenmerk`, `Uw referentie`, `Uw kenmerk`, `Behandeld door`, `Doorkiesnummer`, `Telefoon`, `Bijlage(n)`, `Onderwerp`, `Betreft` in the first 15 lines. These fields are rare in email body text, so the false-positive risk is low.

With `semantic_split=True`, a sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) additionally splits on cosine similarity drops between adjacent pages.

**1d. Finalisation**

Page texts are joined, redaction codes are summed and annotated (`[REDACTED: 5.1.2e]`), and `_categorize_document()` classifies each document into one of 8 categories (Email patterns are scored on the first page only to avoid false positives from quoted headers in body text).

---

#### GPT-4o pipeline (`pipeline_gpt4o.py`)

Three passes, each progressively cheaper and more context-aware.

**Pass 1 — Per-page GPT-4o vision** (`gpt-4o`)

Each page is rendered at 200 DPI, encoded as a base64 JPEG (≤1568 px for optimal `detail=high` tiling), and sent to GPT-4o with a structured JSON prompt. The model returns per page:

| Field | Description |
|---|---|
| `text` | Full page text in reading order; email headers on their own lines; redacted sections as `[REDACTED]` or `<[REDACTED]@domain.nl>` |
| `is_new_document` | Whether this is the first page of a new document |
| `doc_code` | 4-digit WOO stamp visible anywhere on the page, or null |
| `within_doc_page` | Page-within-document counter, or null |
| `category` | Email, Chat, Nota, Brief, Report, Timeline, Vergadernotulen, or Other |
| `is_continuation` | Whether the page text begins mid-sentence (strong continuation signal) |
| `email_start` | Whether a new email header block starts on this page |
| `email_from/to/cc/subject/date` | Extracted from the first new email on this page |

Results stream to a JSON cache file after each page so a crash loses at most one page.

**Pass 2 — Boundary detection** (`gpt-4o-mini`, text-only)

A compact summary of all pages (one line each: stamp, within-doc page, new/continuation flags, category, email flag, and first 500 chars of text) is sent in a single call. The model returns a JSON object listing document start pages and, for email documents, the page numbers where each individual email begins.

This pass corrects pass-1 errors by seeing the full document in context — a page marked `is_new_document=true` by pass 1 can be overruled if the surrounding context makes it clearly a continuation.

**Pass 3 — Full-document email extraction** (`gpt-4o-mini`, text-only, email docs only)

For each document classified as Email, the full concatenated text is sent in one call. The model extracts a structured list of individual emails with subject, sender, recipient, CC, date, and attachment list. This handles Dutch dates, nested reply chains, and redacted headers more reliably than the regex-based `email_splitter`. Falls back to `email_splitter` if the API call fails.

Pass 2 and pass 3 results are written back into the cache JSON so subsequent loads skip both passes.

**Cache and rebuild**

`save_cache()` writes all page metadata after pass 1. `docs_from_cache()` reloads it and re-runs passes 2 and 3 from cache (no vision API calls), enabling full regeneration without cost.

---

### Step 2 — Date extraction and sorting (`text_sorting.py`)

| Category | Strategy |
|---|---|
| E-mail | First parseable `Verzonden`/`Sent` date from the email thread |
| Nota, Brief, Report, Vergadernotulen | `Datum:` field in the first 1500 characters; falls back to date scan |
| Timeline, Other | First `DD-MM-YYYY` or `DD monthname YYYY` pattern in the first 1000 characters |

`_parse_date()` handles Dutch and English date formats including weekday prefixes, glued formats (`17maart2025`), and 12/24h times.

---

### Step 2b — Email splitting (`email_splitter.py`)

`split_emails()` splits an email document's text into individual emails using two boundary signals:

1. A header field type that already appeared in the current email reappears (e.g. a second `Van:` line).
2. A header field appears after 6+ consecutive non-header lines (re-entering a header block from body text).

Outlook `-----Original Message-----` and `-----Oorspronkelijk bericht-----` separators are hard split points. Dutch and English field names are normalised so `Van:` and `From:` are treated as the same field. Redacted addresses are normalised before splitting.

---

### Step 3 — HTML timeline (`visualisation.py`)

`build_html()` produces a single self-contained HTML file.

- **Timeline:** documents laid out chronologically on a horizontal scrollable axis; month labels inserted on change; category filter chips at top.
- **Email documents:** exploded into one card per email, positioned at the email's sent date. Cards show Van/Aan/CC as ministry pill tags (domains resolved to short names), subject, redaction summary, and collapsible body.
- **Non-email documents:** first-page thumbnail; click to open a scrollable page viewer.
- **Redaction display:** bare `5.1.x`/`5.2.x` codes rendered as styled `(REDACTED)` spans; per-email and per-document counts shown.

---

## Evaluation with `annotate.py`

`annotate.py` is a browser-based annotation and evaluation tool. You view the PDF page by page, mark document boundaries as ground truth, and immediately see precision/recall/F1 scores for both pipelines side by side.

### Quick start — auto mode (recommended)

```bash
# First run: runs both pipelines and saves caches
export OPENAI_API_KEY=sk-...
python annotate.py --pdf dossier.pdf --auto

# Every run after: loads from caches (no API cost, no OCR wait)
python annotate.py --pdf dossier.pdf --auto
```

`--auto` loads **GPT-4o as Pipeline A** and **OCR as Pipeline B**. Caches are saved next to the PDF:

| Cache file | Pipeline | To force a re-run |
|---|---|---|
| `dossier_gpt4o_cache.json` | GPT-4o (A) | `rm dossier_gpt4o_cache.json` |
| `dossier_ocr_segs.json` | OCR (B) | `rm dossier_ocr_segs.json` |

### Manual mode

```bash
# OCR only (no GPT-4o)
python annotate.py --pdf dossier.pdf --ocr

# GPT-4o as A, OCR as B explicitly
python annotate.py --pdf dossier.pdf \
  --from-cache dossier_gpt4o_cache.json \
  --compare-segments dossier_ocr_segs.json

# Two GPT-4o cache versions side by side
python annotate.py --pdf dossier.pdf \
  --from-cache cache_v1.json \
  --compare-cache cache_v2.json

# Load existing annotations
python annotate.py --pdf dossier.pdf --auto \
  --annotations annotations_dossier.json
```

### Annotation workflow

1. Open `http://localhost:5050` in a browser
2. Click between pages (or press **B**) to mark a document boundary
3. Select a document in the **GT** list → fill in type and date in the right pane
4. For email documents, expand **Email details** and add one entry per email (subject, sender, date)
5. Press **Tab** to jump to the next unannotated document
6. Metrics update live — Precision / Recall / F1 shown for both pipelines
7. Click **Export JSON** to save the full annotation + metrics file

Annotations auto-save to `annotations_dossier.json` next to the PDF.

### What the metrics mean

| Metric | Definition |
|---|---|
| Boundary F1 | F1 score on document-start page predictions vs ground truth |
| Boundary P / R | Precision (no false splits) / Recall (no missed splits) |
| Email subj. F1 | Fuzzy subject match between pipeline and GT emails (`pip install rapidfuzz`) |
| Email date acc. | Exact ISO date match for subject-matched email pairs |
| Email sender acc. | Fuzzy sender match for subject-matched email pairs |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `B` | Toggle GT boundary at current page |
| `Tab` | Jump to next unannotated document |
| `1`–`9` | Set document type (E-mail, Nota, Brief, …) |
| `↑` / `↓` | Scroll one page |
| `S` | Force save |

### All CLI flags

```
--pdf FILE               PDF to annotate (required)
--auto                   Auto mode: GPT-4o (A) + OCR (B), cached
--api-key KEY            OpenAI key (falls back to OPENAI_API_KEY)
--from-cache FILE        Pipeline A: GPT-4o cache JSON
--from-segments FILE     Pipeline A: pre-saved segments JSON
--ocr                    Pipeline A: run OCR
--compare-cache FILE     Pipeline B: GPT-4o cache JSON
--compare-segments FILE  Pipeline B: pre-saved segments JSON
--compare-ocr            Pipeline B: run OCR
--compare-api-key KEY    Pipeline B: OpenAI key
--annotations FILE       Load existing annotations JSON
--port PORT              HTTP port (default: 5050)
```

---

## File overview

| File | Role |
|---|---|
| `main.py` | CLI entry point — OCR pipeline |
| `main_gpt4o.py` | CLI entry point — GPT-4o pipeline |
| `server.py` | FastAPI server — WOOLens browser UI + `/api/analyse` SSE endpoint |
| `pipeline_ocr.py` | Text extraction, stamp detection, OCR, document splitting |
| `pipeline_gpt4o.py` | GPT-4o three-pass extraction, boundary detection, email parsing, caching |
| `pipeline_inventarislijst.py` | GPT-4o-mini extraction of WOO inventory table (inventarislijst) |
| `annotate.py` | Browser-based annotation and dual-pipeline evaluation tool |
| `text_sorting.py` | Date extraction and chronological sorting |
| `email_splitter.py` | Splits email document text into individual emails |
| `visualisation.py` | Interactive HTML timeline generation |
| `diagnose_stamps.py` | Diagnostic — inspect stamp detection per page and region |

---

## Output schema

Both pipelines return `dict[str, dict]` keyed by document code:

| Field | Type | Description |
|---|---|---|
| `doc_code` | `str` | 4-digit stamp, `unknown_N`, or `auto_NNN` |
| `pages` | `list` | PIL images (GPT-4o) or 1-based page numbers (OCR) |
| `page_nums_in_doc` | `list[int\|None]` | Within-document page stamps |
| `text` | `str` | Full extracted text |
| `annotated_text` | `str` | Text with `[REDACTED: 5.1.2e]` markers |
| `redaction_codes` | `dict[str, int]` | WOO redaction grounds and counts |
| `category` | `str` | E-mail, Chat, Nota, Brief, Report, Timeline, Vergadernotulen, or Other |
| `method` | `str` | `direct`, `ocr`, `gpt4o-stamp`, or `gpt4o-boundary` |
| `emails` | `list[dict]` | Structured per-email data (GPT-4o pipeline, email docs only) |
