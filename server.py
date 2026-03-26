"""
WOOLens + WOO Organizer unified server.

Serves the frontend and proxies requests to Dutch government WOO databases.
Also exposes /api/analyse — a server-side pipeline endpoint (SSE) that runs
either the OCR pipeline (pdfplumber + tesseract) or the GPT-4o vision pipeline
on a PDF and returns structured emails + timeline JSON.

Run:
    python server.py
    # or: uvicorn server:app --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

app = FastAPI(title="WOOLens + WOO Organizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_HOSTS = {"pid.wooverheid.nl", "open.overheid.nl"}
INDEX_HTML = Path(__file__).parent / "index.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WOOLens/1.0; +https://woozm.nl)"}

# Prevent concurrent pipeline runs — stdout capture is process-wide.
_pipeline_lock = threading.Lock()

# Last uploaded PDF kept for in-app page preview.
_SESSION_PDF: Path = Path(tempfile.gettempdir()) / "woo_session.pdf"


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return INDEX_HTML.read_text(encoding="utf-8")


# ── /proxy?url=XXX ───────────────────────────────────────────────────────────

@app.get("/proxy")
async def proxy(url: str = Query(..., description="Full URL to proxy")):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail=f"Host not allowed: {parsed.hostname}")
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="Only https URLs are allowed")

    async def _stream():
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            async with client.stream("GET", url, headers=HEADERS) as r:
                async for chunk in r.aiter_bytes(chunk_size=65536):
                    yield chunk

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        head = await client.head(url, headers=HEADERS)
        content_type = head.headers.get("content-type", "application/octet-stream")

    return StreamingResponse(_stream(), media_type=content_type)


# ── /api/session_pdf & /api/session_pdf_range ────────────────────────────────

@app.get("/api/session_pdf")
async def session_pdf():
    """Serve the most recently analysed PDF for in-app page preview."""
    if not _SESSION_PDF.exists():
        raise HTTPException(status_code=404, detail="No session PDF available")
    return FileResponse(
        _SESSION_PDF,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/api/session_pdf_range")
async def session_pdf_range(
    start: int = Query(..., description="First page (1-indexed, inclusive)"),
    end:   int = Query(..., description="Last page (1-indexed, inclusive)"),
):
    """Extract a page range from the session PDF and return it as a new PDF."""
    if not _SESSION_PDF.exists():
        raise HTTPException(status_code=404, detail="No session PDF available")
    import io
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(_SESSION_PDF))
    writer = PdfWriter()
    for page_num in range(start - 1, min(end, len(reader.pages))):
        writer.add_page(reader.pages[page_num])
    buf = io.BytesIO()
    writer.write(buf)
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


# ── /infobox?pid=XXX ─────────────────────────────────────────────────────────

@app.get("/infobox")
async def infobox(pid: str = Query(...)):
    target = f"https://pid.wooverheid.nl/?pid={pid}&infobox=true"
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        r = await client.get(target, headers=HEADERS)
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail="Upstream error")
    return Response(content=r.content, media_type="application/json")


# ── /text?pid=XXX ────────────────────────────────────────────────────────────

@app.get("/text")
async def text(pid: str = Query(...)):
    target = f"https://pid.wooverheid.nl/?pid={pid}&text=true"
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        r = await client.get(target, headers=HEADERS)
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail="Upstream error")
    return Response(
        content=json.dumps({"text": r.text}),
        media_type="application/json",
    )


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_title(doc: dict) -> str:
    """Best-effort title for a document (email subject or first line)."""
    if doc.get("category") == "E-mail":
        try:
            from email_splitter import split_emails
            emails = split_emails(doc["text"], doc.get("doc_code", ""))
            if emails and emails[0].get("subject"):
                return emails[0]["subject"]
        except Exception:
            pass
    for line in doc.get("text", "").split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 5:
            return stripped[:120]
    return doc.get("doc_code", "Document")


def _extract_sender(doc: dict) -> str:
    """Best-effort sender from the first Van:/From: line."""
    m = re.search(r"(?i)(?:van|from)\s*:\s*(.+)", doc.get("text", "")[:1000])
    return m.group(1).strip()[:80] if m else ""


_HEADER_LINE_RE = re.compile(
    r"^(?:van|aan|from|to|cc|bcc|datum|onderwerp|subject|verzonden|sent)\s*:",
    re.IGNORECASE,
)

def _generate_description(doc: dict) -> str:
    """Short human-readable description of a document for display in cards."""
    category = doc.get("category", "Other")
    text = doc.get("text", "")
    redaction_codes = doc.get("redaction_codes", {})

    # Email: use subject of first email
    if category == "E-mail":
        try:
            from email_splitter import split_emails
            emails = split_emails(doc.get("annotated_text") or text, doc.get("doc_code", ""))
            if emails and emails[0].get("subject"):
                return emails[0]["subject"][:100]
        except Exception:
            pass
        return "E-mail"

    # For other types, find the first meaningful non-header line
    for line in text.split("\n")[:20]:
        stripped = line.strip()
        if len(stripped) > 8 and not _HEADER_LINE_RE.match(stripped):
            # Skip lines that are purely digits/codes
            if not re.fullmatch(r"[\d\s\.\-\/]+", stripped):
                return stripped[:100]

    # Fallback: mention redaction codes if any
    if redaction_codes:
        top_codes = ", ".join(list(redaction_codes.keys())[:3])
        return f"Geredigeerd document ({top_codes})"

    return category


def _pipeline_to_json(docs: dict) -> dict:
    """
    Convert pipeline output (dict[str, dict]) to frontend-ready JSON.
    Calls text_sorting.sort_documents and email_splitter.split_emails.
    PIL Image objects in 'pages' are intentionally excluded.
    """
    from text_sorting import sort_documents
    from email_splitter import split_emails

    sorted_docs = sort_documents(docs)

    emails: list[dict] = []
    timeline: list[dict] = []
    all_redaction: dict[str, int] = {}

    for code, doc in sorted_docs.items():
        dt = doc.get("date")
        date_str = dt.strftime("%Y-%m-%d") if dt else ""
        category = doc.get("category", "Other")

        preview_text = (doc.get("annotated_text") or doc.get("text", ""))[:800]
        # First actual PDF page for this document (int for OCR, tracked separately for GPT-4o)
        raw_pages = doc.get("pages", [])
        pdf_pages = doc.get("pdf_pages") or (raw_pages if raw_pages and isinstance(raw_pages[0], int) else [])
        pdf_page_start = pdf_pages[0]  if pdf_pages else None
        pdf_page_end   = pdf_pages[-1] if pdf_pages else None
        timeline.append({
            "id":             code,
            "type":           category,
            "date":           date_str,
            "title":          _extract_title(doc),
            "description":    _generate_description(doc),
            "sender":         doc.get("doc_sender") or _extract_sender(doc),
            "preview":        preview_text,
            "pdf_page_start": pdf_page_start,
            "pdf_page_end":   pdf_page_end,
        })

        if category == "E-mail":
            try:
                # Always run both splitters; use whichever finds more individual emails.
                # Enrich text-based results with GPT-4o metadata (subject/sender/date)
                # when the text splitter finds more splits.
                gpt_emails = doc.get("emails") or []
                try:
                    text_emails = split_emails(doc.get("annotated_text") or doc["text"], code)
                except Exception:
                    text_emails = []
                if len(text_emails) > len(gpt_emails):
                    for i, em in enumerate(text_emails):
                        if i < len(gpt_emails):
                            g = gpt_emails[i]
                            if not em.get("subject") and g.get("subject"): em["subject"] = g["subject"]
                            if not em.get("sender")  and g.get("sender"):  em["sender"]  = g["sender"]
                            if not em.get("date")    and g.get("date"):    em["date"]    = g["date"]
                    src_emails = text_emails
                elif gpt_emails:
                    src_emails = gpt_emails
                else:
                    src_emails = text_emails
                for em in src_emails:
                    emails.append({
                        "id":          em.get("id", f"{code}.?"),
                        "subject":     em.get("subject")     or "",
                        "sender":      em.get("sender")      or "",
                        "to":          em.get("to")          or "",
                        "cc":          em.get("cc")          or "",
                        "date":        em.get("date")        or "",
                        "attachments": em.get("attachments") or [],
                        "text":        em.get("text")        or "",
                    })
            except Exception:
                pass

        for rc, count in doc.get("redaction_codes", {}).items():
            all_redaction[rc] = all_redaction.get(rc, 0) + count

    total_pages = sum(len(d.get("pages", [])) for d in docs.values())

    return {
        "emails":   emails,
        "timeline": timeline,
        "stats": {
            "docs":           len(docs),
            "pages":          total_pages,
            "redactionCodes": all_redaction,
        },
    }


# ── SSE pipeline runner ───────────────────────────────────────────────────────

async def _run_pipeline_sse(
    pipeline: str,
    pdf_path: Path,
    api_key: str | None,
    page_range: tuple[int, int] | None = None,
) -> AsyncIterator[str]:
    """
    Async generator — runs the chosen pipeline in a background thread,
    forwards captured stdout lines as SSE progress events, then emits
    a final 'done' event with the serialised result.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[dict | None] = asyncio.Queue()

    class _StdoutCapture:
        """Thread-safe stdout replacement that forwards lines to the asyncio queue."""
        def __init__(self) -> None:
            self._buf = ""

        def write(self, text: str) -> None:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    asyncio.run_coroutine_threadsafe(
                        q.put({"type": "log", "msg": line}), loop
                    )

        def flush(self) -> None:
            pass

    def _run_in_thread() -> None:
        if not _pipeline_lock.acquire(blocking=False):
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "msg": "Er loopt al een pipeline. Wacht tot deze klaar is of herstart de server."}), loop
            )
            asyncio.run_coroutine_threadsafe(q.put(None), loop)
            return
        old_stdout = sys.stdout
        capture = _StdoutCapture()
        try:
            sys.stdout = capture
            if pipeline == "ocr":
                from pipeline_ocr import load_pdf
                result = load_pdf(pdf_path, page_range=page_range)
            else:
                from pipeline_gpt4o import load_pdf_vlm
                result = load_pdf_vlm(pdf_path, api_key=api_key, page_range=page_range)
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "result", "data": result}), loop
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "msg": str(exc)}), loop
            )
        finally:
            sys.stdout = old_stdout
            _pipeline_lock.release()
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    yield _sse({"type": "progress", "msg": "Pipeline gestart…", "pct": 3})

    result_data: dict | None = None
    while True:
        item = await q.get()
        if item is None:
            break
        if item["type"] == "log":
            yield _sse({"type": "progress", "msg": item["msg"], "pct": None})
        elif item["type"] == "result":
            result_data = item["data"]
        elif item["type"] == "error":
            yield _sse({"type": "error", "msg": item["msg"]})
            return

    if result_data is not None:
        yield _sse({
            "type": "progress",
            "msg":  "Documenten sorteren en e-mails extraheren…",
            "pct":  93,
        })
        try:
            output = _pipeline_to_json(result_data)
            output["pdf_url"] = "/api/session_pdf"
            yield _sse({"type": "done", **output})
        except Exception as exc:
            yield _sse({"type": "error", "msg": f"Resultaten verwerken mislukt: {exc}"})


async def _pipeline_with_cleanup(
    pipeline: str,
    pdf_path: Path,
    api_key: str | None,
    page_range: tuple[int, int] | None = None,
) -> AsyncIterator[str]:
    """Wraps _run_pipeline_sse to delete the temp PDF when done.
    Saves a copy to _SESSION_PDF first so the frontend can display PDF pages."""
    shutil.copy2(pdf_path, _SESSION_PDF)
    try:
        async for chunk in _run_pipeline_sse(pipeline, pdf_path, api_key, page_range=page_range):
            yield chunk
    finally:
        pdf_path.unlink(missing_ok=True)


# ── /api/analyse ─────────────────────────────────────────────────────────────

@app.post("/api/analyse")
async def analyse(
    pipeline:   str       = Form(..., description="'ocr' or 'gpt4o'"),
    url:        str | None = Form(None, description="PDF URL (for online dossiers)"),
    api_key:    str | None = Form(None, description="OpenAI API key (gpt4o only)"),
    file:       UploadFile | None = File(None, description="Uploaded PDF file"),
    page_start: int | None = Form(None, description="First page to process (1-indexed, inclusive)"),
    page_end:   int | None = Form(None, description="Last page to process (1-indexed, inclusive)"),
):
    """
    Run the OCR or GPT-4o pipeline on a PDF and stream progress + results via SSE.

    Accepts either:
      - multipart field 'file' (uploaded PDF), or
      - form field 'url'       (PDF downloaded via proxy)

    Returns text/event-stream with JSON events:
      {"type": "progress", "msg": "...", "pct": 0-100 | null}
      {"type": "done",     "emails": [...], "timeline": [...], "stats": {...}}
      {"type": "error",    "msg": "..."}
    """
    if pipeline not in ("ocr", "gpt4o"):
        raise HTTPException(
            status_code=400,
            detail="pipeline must be 'ocr' or 'gpt4o'",
        )

    # ── Obtain PDF bytes ──────────────────────────────────────────────────────
    if file is not None:
        pdf_bytes = await file.read()
    elif url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                r = await client.get(url, headers=HEADERS)
            if not r.is_success:
                raise HTTPException(
                    status_code=502,
                    detail=f"Kon PDF niet ophalen: HTTP {r.status_code}",
                )
            pdf_bytes = r.content
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"PDF ophalen mislukt: {exc}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Geef 'file' of 'url' op",
        )

    # ── Write to temp file ────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.close()
    pdf_path = Path(tmp.name)

    page_range = (page_start, page_end) if page_start and page_end else None

    return StreamingResponse(
        _pipeline_with_cleanup(pipeline, pdf_path, api_key or None, page_range=page_range),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /api/inventarislijst ─────────────────────────────────────────────────────

@app.post("/api/inventarislijst")
async def analyse_inventarislijst(
    api_key:    str            = Form(...,  description="OpenAI API key for GPT-4o-mini"),
    file:       UploadFile | None = File(None, description="Separate Inventarislijst PDF"),
    page_start: int | None    = Form(None, description="First page in session PDF (1-indexed)"),
    page_end:   int | None    = Form(None, description="Last page in session PDF (1-indexed)"),
):
    """
    Extract the document inventory table from a WOO Inventarislijst.

    Accepts either:
      - multipart 'file' — a dedicated Inventarislijst PDF upload
      - no file + page_start/page_end — a page range within the current session PDF
      - no file + no range — the entire session PDF

    Returns JSON: {"items": [...], "total": N}
    Each item: {code, title, date, pages, decision, grounds}
    """
    if file is not None:
        pdf_bytes = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()
        pdf_path   = Path(tmp.name)
        is_tmp     = True
        page_range = None
    elif _SESSION_PDF.exists():
        pdf_path   = _SESSION_PDF
        is_tmp     = False
        page_range = (page_start, page_end) if page_start and page_end else None
    else:
        raise HTTPException(
            status_code=400,
            detail="Geen PDF beschikbaar. Upload eerst een dossier of geef een apart bestand op.",
        )

    try:
        from pipeline_inventarislijst import extract_inventarislijst
        items = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: extract_inventarislijst(pdf_path, api_key, page_range=page_range),
        )
        return {"items": items, "total": len(items)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if is_tmp:
            pdf_path.unlink(missing_ok=True)


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
