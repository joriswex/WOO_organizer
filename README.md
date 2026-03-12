# WOO Organizer

A pipeline that converts Dutch WOO (*Wet Open Overheid*) disclosure PDFs into an interactive HTML timeline — with automatic document splitting, date extraction, email thread rendering, attachment tags, recipient resolution, and redaction-code tracking.

---

## What it does

1. **Loads a PDF** — either text-layer (pdfplumber) or image-based (OCR via Tesseract)
2. **Splits the PDF into individual documents** — using page stamps, email headers, or GPT-4o boundary detection
3. **Extracts dates** — assigns each document to a point on the timeline
4. **Categorises each document** — E-mail, Nota, Brief, Report, Chat, Timeline, Vergadernotulen, Other
5. **Renders an interactive HTML timeline** — with thumbnails, full text, email thread view, chat bubble view, attachment tags, recipient pills, and per-email redaction-code summaries

---

## Pipelines

There are two independent pipelines that share the same output format and visualisation layer.

### Standard OCR pipeline (`main.py`)

Uses pdfplumber for searchable PDFs and Tesseract OCR for image-based pages. Fast and runs fully offline. Best for PDFs with clear machine-readable stamps.

```bash
python main.py                                          # test.pdf → woo_timeline.html
python main.py --pdf my_dossier.pdf --out output.html
```

### GPT-4o VLM pipeline (`main_gpt4o.py`)

Sends every page to GPT-4o vision for full text extraction and classification. Produces cleaner output on PDFs with heavy redaction, complex layouts, or no machine-readable stamps. Results are cached to JSON so the HTML can be regenerated without re-calling the API.

```bash
python main_gpt4o.py --pdf my_dossier.pdf               # full run, saves cache automatically
python main_gpt4o.py --pdf my_dossier.pdf --vlm-pages 15  # test batch: first 15 pages only
python main_gpt4o.py --from-cache my_dossier_gpt4o_cache.json --pdf my_dossier.pdf
```

---

## Setup

**Requirements:** Python 3.11+, [Tesseract](https://github.com/tesseract-ocr/tesseract), [Poppler](https://poppler.freedesktop.org/) (for pdf2image)

```bash
# Install system dependencies (macOS)
brew install tesseract poppler

# Create virtual environment and install Python packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**API key** (only needed for the GPT-4o pipeline):

```bash
export OPENAI_API_KEY=sk-...
```

---

## File overview

| File | Role |
|---|---|
| `main.py` | Entry point — standard OCR pipeline |
| `main_gpt4o.py` | Entry point — GPT-4o VLM pipeline |
| `pipeline_ocr.py` | PDF loading, stamp detection, OCR, document splitting |
| `pipeline_gpt4o.py` | GPT-4o full-page text extraction and classification |
| `text_sorting.py` | Date extraction and chronological sorting |
| `visualisation.py` | Interactive HTML timeline generation |
| `email_splitter.py` | Splits email document text into individual emails |
| `diagnose_stamps.py` | Diagnostic tool — inspect stamp detection per page |

---

## Document splitting logic

The OCR pipeline uses a two-stage approach:

1. **Stage 1** — Per-page: detect 4-digit document code stamps (text layer + raster OCR), detect within-document page numbers. Searches all corners/edges to support different WOO layout formats.
2. **Stage 2** — If stamps found: forward-fill codes across pages, create `unknown_N` slots at readable boundaries. If no stamps found: auto-split at email headers, "Pagina 1 van" markers, or semantic boundaries → assigns `auto_001`, `auto_002`, …

The GPT-4o pipeline combines both stages into a single API call per page.

---

## Output format

Both pipelines return `dict[str, dict]` keyed by document code. Each document contains:

| Field | Type | Description |
|---|---|---|
| `doc_code` | `str` | 4-digit stamp code, `unknown_N`, or `auto_NNN` |
| `pages` | `list[Image]` | PIL page images |
| `page_nums_in_doc` | `list[int\|None]` | Within-document page numbers |
| `text` | `str` | Full extracted text |
| `annotated_text` | `str` | Text with `[REDACTED: 5.1.2e]` markers |
| `redaction_codes` | `dict[str, int]` | WOO redaction codes and their counts |
| `category` | `str` | Document type |
| `method` | `str` | Extraction method used |

The GPT-4o pipeline additionally sets `chat_name` and `chat_messages` on Chat documents.

---

## Benchmarking

Both pipelines produce the same output schema, making them directly comparable. Run both on the same PDF and open the two HTML files side by side:

```bash
python main.py --pdf dossier.pdf --out timeline_ocr.html
python main_gpt4o.py --pdf dossier.pdf --out timeline_gpt4o.html
```

The GPT-4o cache (`*_gpt4o_cache.json`) preserves the raw per-page extraction so the HTML can be regenerated without additional API costs.
