# WOO Organizer

Converts Dutch WOO (*Wet Open Overheid*) disclosure PDFs into an interactive HTML timeline. Documents are split, dated, categorised, and rendered with inline email cards, chat threads, page thumbnails, redaction-code summaries, and inventory-list validation.

---

## Architecture overview

The project consists of a FastAPI server (`server.py`) that serves the frontend (`index.html`) and exposes a single pipeline endpoint (`/api/analyse`). When a PDF is uploaded, the server runs either the OCR pipeline or the GPT-4o vision pipeline in a background thread and streams progress and results back to the browser via Server-Sent Events.

```
browser  ←→  server.py (FastAPI)
                ├── GET  /                         → index.html
                ├── GET  /proxy?url=…              → proxies WOO database requests
                ├── GET  /api/session_pdf[_range]  → serves the last uploaded PDF
                ���── POST /api/analyse              → runs pipeline, streams SSE
                        ├── pipeline_ocr.py        (pipeline="ocr")
                        └── pipeline_gpt4o.py      (pipeline="gpt4o")
                                └── pipeline_inventarislijst.py  (Inventarislijst pages)
```

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

## Running the server

```bash
python server.py
# or: uvicorn server:app --port 8000 --reload
```

Open `http://localhost:8000` in a browser. Upload a WOO dossier PDF, choose a pipeline, and click Analyseer.

The GPT-4o pipeline requires an OpenAI API key. This can be set as an environment variable (`OPENAI_API_KEY`) or entered directly in the UI.

---

## Pipelines

Two independent pipelines share the same date-extraction and sorting layer. They differ only in how they extract text and detect document boundaries.

| Pipeline | `pipeline` value | Text extraction | Requires |
|---|---|---|---|
| OCR | `"ocr"` | pdfplumber (searchable PDFs) + Tesseract (image-based) | Tesseract, Poppler |
| GPT-4o VLM | `"gpt4o"` | GPT-4o vision API, one call per page | OpenAI API key |

Both produce the same output schema and are directly comparable.

---

## Pipeline walkthrough

### Step 1 — Text extraction and document splitting

This is the only step that differs between the two pipelines.

---

#### OCR pipeline (`pipeline_ocr.py`)

**1a. Searchability check**

`_is_searchable()` opens the PDF with pdfplumber and checks whether any page contains selectable text. The result decides whether the pipeline uses direct text extraction (fast) or full raster OCR (slower, handles image-based scans).

**1b. Stage 1 — Per-page data collection**

For every page the pipeline collects:

**Doc code** — the 4-digit WOO catalogue stamp (e.g. `0144`) that identifies which document a page belongs to.

- *Text layer:* `_find_doc_code_words()` scans the top strip using pdfplumber's word list. Only top regions are searched (`_DOC_CODE_TEXT_REGIONS`: top-right → top-left → top-full) to avoid false positives from body text. Year numbers (1900–2099) are excluded.
- *Raster OCR:* `_find_doc_code_raster()` renders the page at 200 DPI and OCRs six corner/edge regions in priority order (`_DOC_CODE_RASTER_REGIONS`: top-right → top-left → bottom-right → bottom-middle → bottom-left → top-full). Two-stage matching per region:
  1. Primary: `\b(0\d{3})\b` — requires a leading zero; rejects article-number fragments like `5122`.
  2. Fallback (bottom regions only): `\d{2,}(0\d{3})` — extracts embedded codes from barcode strings like `7601430` → `0143`.

**Within-doc page number** — `_find_within_doc_page_raster()` OCRs a tight strip at the bottom-right corner and returns the per-document page counter (e.g. page 1 of 3).

**Page text** — `page.extract_text()` for searchable PDFs; `_ocr_image()` (Tesseract, Dutch+English) for image-based pages.

**Redaction codes** — WOO legal grounds (regex `5\.[12]\.[1-9][a-z]{0,2}`, e.g. `5.1.2e`) are counted per page. An optional supplementary 300 DPI OCR pass captures image-only redaction stamps.

**1c. Stage 2 — Document code assignment**

- **Stamps detected** → `_build_docs_forward_fill()`: each page inherits the most recently seen stamp. Pages without a stamp but with `within_doc_page == 1` or `"Pagina 1 van"` in the text get an auto-generated `unknown_N` code.
- **No stamps anywhere** → `_build_docs_auto_split()`: boundaries are detected from `within_doc_page == 1`, `"Pagina 1 van"` text, or a fresh email header block appearing after non-email content. Documents receive `auto_001`, `auto_002`, … codes. With `semantic_split=True`, a multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`) also detects cosine-similarity drops between adjacent pages.

**1d. Finalisation**

Page texts are joined, redaction codes are summed and annotated as `[REDACTED: 5.1.2e]` markers, and `_categorize_document()` classifies each document by keyword scoring against 8 category rule sets (Email patterns scored on first page only to prevent false positives from quoted headers in the body).

---

#### GPT-4o pipeline (`pipeline_gpt4o.py`)

**Pipeline architecture**

| Step | Model | Input | Purpose |
|---|---|---|---|
| Raster survey | Tesseract (free) | All page images | Detect stamp format, coverage, and sequence before any API call |
| Pass 0 (pilot) | gpt-4o | 8 sampled pages | Confirm stamp format (4-digit, DocNr, 7-digit barcode, none); skipped for stampless dossiers |
| Pass 1 | gpt-4o | All page images (base64 JPEG, 200 DPI) | Per-page text, category, doc code, boundary signals, email/chat metadata |
| Pass 2 | gpt-4o-mini | Concatenated page texts | Document boundary detection for unstamped dossiers |
| Pass 3 | gpt-4o-mini | Full document text (E-mail docs only) | Structured email extraction from the complete thread |

**Raster stamp survey (pass-0b)**

Before any GPT-4o call, `_raster_stamp_survey()` runs Tesseract OCR on all pages in parallel to determine whether the dossier uses stamps:

- Checks **monotonicity** of the detected code sequence — real stamps increase page-by-page; random OCR artefacts bounce randomly. A sequence with fewer than 60% non-decreasing consecutive pairs is treated as noise.
- Requires at least 2 distinct valid codes and ≥10% page coverage to classify a dossier as stamped.
- Detects **large jumps** (gap > 5 between sustained runs) as likely document boundaries where code numbers were skipped.
- Derives the expected code range from sustained runs only (runs of ≥2 pages), ignoring single-page OCR outliers.
- The result is injected into the GPT-4o pass-1 system prompt as a per-page cross-check table and jump list.

For stampless dossiers (near-zero raster coverage), the GPT-4o pilot pass is skipped entirely. For dossiers with sparse but nonzero coverage (e.g. DocNr-format where labels appear only on first pages), the pilot still runs.

**Stamp detection hardening**

- `_normalise_doc_code()` rejects `0000` — it is not a valid WOO document code (valid range `0001`–`9999`).
- `_ocr_stamp_fallback()` requires at least 2 distinct seed codes before spreading stamps to unstamped pages via OCR. A single repeated false-positive code is not sufficient to trigger recovery.

**Pass 1 — Per-page analysis**

Each page is rendered at 200 DPI, resized to ≤1568 px (optimal for `detail=high`), and sent as a base64 JPEG. The system prompt includes the raster survey cross-check and a dossier-specific format hint from the pilot. The model returns a structured JSON object per page:

| Field | Description |
|---|---|
| `text` | Full page text in reading order; email headers on their own lines |
| `is_new_document` | Whether this is the first page of a new document |
| `doc_code` | Stamp code visible on the page (normalised to 4 digits), or null |
| `within_doc_page` | Per-document page counter (`"Pagina X van N"`), or null |
| `category` | E-mail, Chat, Nota, Brief, Report, Timeline, Vergadernotulen, or Other |
| `doc_subtype` | Fine-grained subtype (e.g. `kamerbrief`, `besluit`, `chat_sms`) |
| `chat_name` | Chat group or contact name (Chat pages only) |
| `chat_messages` | Array of `{sender_position, sender_label, timestamp, content}` (Chat pages only) |
| `email_start`, `email_from/to/cc/subject/date` | Per-page email header fields |

After pass 1, per-page `raster_code` values from the survey are attached to each page. `_stamp_coverage()` and `_build_docs_forward_fill()` use `raster_code` as a fallback when GPT-4o returned no `doc_code`.

Results are saved to a cache file (`<stem>_gpt4o_cache.json`) after pass 1. Pass 1 is never re-run if the cache already exists.

**Pass 2 — Boundary detection**

Only runs for dossiers where stamp coverage is below the reliability threshold. The page text summary is sent to gpt-4o-mini. If the dossier contains an **inventarislijst** (document inventory table), its document list is extracted and prepended to the boundary-detection prompt — giving the model explicit document titles to match against email subjects and letter headings, significantly improving recall.

**Pass 3 — Email extraction**

For each document classified as E-mail, the full concatenated text is sent to gpt-4o-mini in one call. It returns structured email objects with subject, sender, recipients, date, time, attachments, and body. Results are stored in `emails_by_doc` in the cache.

**Caching**

The cache JSON stores all pass-1 per-page data, pass-2 boundary decisions, and pass-3 email results. A cached dossier can be fully re-analysed or re-displayed with zero API calls.

---

#### Inventarislijst pipeline (`pipeline_inventarislijst.py`)

Extracts the document inventory table from a WOO *inventarislijst* page using GPT-4o-mini vision. Called automatically when the main pipeline detects an Inventarislijst document. Returns a list of rows with `code`, `title`, `date`, `pages`, `decision` (openbaar / gedeeltelijk openbaar / niet openbaar), and `grounds` (WOO redaction articles).

---

### Step 2 — Date extraction and sorting (`text_sorting.py`)

For each document a date is extracted using a category-specific strategy:

| Category | Strategy |
|---|---|
| E-mail | Date from the first parseable `Verzonden`/`Sent` header in the email thread |
| Nota, Brief, Report, Vergadernotulen | Searches first 1 500 characters for a `Datum:` field; falls back to text scan |
| Timeline, Other, Chat | Scans first 1 000 characters for any `DD-MM-YYYY` or `DD monthname YYYY` pattern |

`_parse_date()` handles: `DD-MM-YYYY`, `DD monthname YYYY`, Dutch weekday + date (`maandag 17 maart 2025`), English long date, US format with weekday (`Mon 3/17/2025`), and glued formats (`17maart2025`). Both Dutch and English month names are recognised.

Documents are sorted chronologically. Those without a parseable date are appended at the end in doc-code order.

---

### Step 3 — Email splitting (`email_splitter.py`)

`split_emails()` splits the concatenated text of an email document into individual emails. Boundaries are detected by:

1. A header field type that already appeared in the current email reappears (e.g. a second `Van:` line).
2. A header field appears after 6 or more consecutive non-header lines.
3. Outlook separator lines (`---- Oorspronkelijk bericht ----`, `---- Original Message ----`).

Both Dutch (`Van`, `Aan`, `Onderwerp`, `Verzonden`, `CC`) and English header field names are normalised before comparison. Each email gets an ID like `0003.2` with `subject`, `sender`, `to`, `cc`, `date` (YYYY-MM-DD), and `time` (HH:MM) fields.

---

## File overview

| File | Role |
|---|---|
| `server.py` | FastAPI server — serves the frontend, proxies WOO database requests, exposes `/api/analyse` SSE endpoint |
| `index.html` | WOOLens frontend — single-file vanilla JS + CSS, no build step; uses PDF.js for in-browser page rendering |
| `pipeline_ocr.py` | OCR pipeline: stamp detection, pdfplumber + Tesseract text extraction, document splitting |
| `pipeline_gpt4o.py` | GPT-4o pipeline: three-pass vision extraction, boundary detection, email extraction, cache management |
| `pipeline_inventarislijst.py` | GPT-4o-mini vision extraction of WOO inventory table rows |
| `email_splitter.py` | Splits email document text into individual emails with structured metadata |
| `text_sorting.py` | Date extraction and chronological sorting |
| `requirements.txt` | Python dependencies |
| `annotate.py` | Browser-based annotation tool for ground-truth labelling and pipeline comparison (development) |
| `evaluate_pipelines.py` | Offline evaluation script — compares OCR vs GPT-4o on boundary detection and type classification (development) |

---

## Output schema

Both pipelines return `dict[str, dict]` keyed by document code. Each document contains:

| Field | Type | Description |
|---|---|---|
| `doc_code` | `str` | 4-digit stamp code, `unknown_N`, or `auto_NNN` |
| `pages` | `list` | PIL page images (GPT-4o) or 1-based page numbers (OCR) |
| `page_nums_in_doc` | `list[int\|None]` | Within-document page stamps |
| `text` | `str` | Full extracted text |
| `annotated_text` | `str` | Text with `[REDACTED: 5.1.2e]` markers |
| `redaction_codes` | `dict[str, int]` | WOO redaction codes and their counts |
| `category` | `str` | E-mail, Chat, Nota, Brief, Report, Timeline, Vergadernotulen, or Other |
| `doc_subtype` | `str` | Fine-grained subtype (e.g. `kamerbrief`, `chat_sms`); `other` if undetermined |
| `method` | `str` | `direct`, `ocr`, `gpt4o-stamp`, or `gpt4o-boundary` |
| `emails` | `list[dict]` | Structured emails (E-mail documents, GPT-4o pipeline) |
| `chat_messages` | `list[dict]` | Structured chat messages (Chat documents, GPT-4o pipeline) |

---

## `/api/analyse` endpoint reference

```
POST /api/analyse
Content-Type: multipart/form-data

pipeline:   "ocr" | "gpt4o"           (required)
file:       <PDF upload>               (or use url)
url:        <PDF URL>                  (or use file)
api_key:    <OpenAI key>               (gpt4o only; falls back to OPENAI_API_KEY env var)
page_start: <int>                      (optional — first page, 1-indexed)
page_end:   <int>                      (optional — last page, 1-indexed)
```

Returns `text/event-stream` with JSON events:

```json
{"type": "progress", "msg": "...", "pct": 0–100}
{"type": "done",     "emails": [...], "timeline": [...], "stats": {...}}
{"type": "error",    "msg": "..."}
```

---

## WOO redaction codes

Legal grounds for redaction follow the pattern `5.[12].[1-9][a-z]{0,2}` (e.g. `5.1.2e`, `5.2.1`). They are:

- Counted per document and stored in `redaction_codes`
- Marked as `[REDACTED: 5.1.2e]` in `annotated_text`
- Rendered as styled `(REDACTED)` spans in the frontend email body view
- Summarised per document and per email in the timeline UI
