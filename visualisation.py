"""
visualisation.py — generate woo_timeline.html from sorted WOO documents.

Produces a single self-contained HTML file with:
  - Horizontal scrollable timeline, one card per WOO document
  - Pillow-rendered text thumbnails embedded as base64 PNGs
  - Clickable side panel showing full document / email thread
"""
from __future__ import annotations

import base64
import html as html_mod
import io
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from email_splitter import split_emails

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_PATH = Path("test.pdf")
OUT_PATH = Path("woo_timeline.html")

THUMB_W, THUMB_H = 200, 280

CATEGORY_COLORS = {
    "E-mail":          "#2563EB",
    "Nota":            "#16A34A",
    "Report":          "#EA580C",
    "Timeline":        "#7C3AED",
    "Vergadernotulen": "#0891B2",
    "Brief":           "#CA8A04",
    "Other":           "#6B7280",
}
DEFAULT_COLOR = "#6B7280"


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# Thumbnail renderer
# ---------------------------------------------------------------------------
def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                  width: int, font: ImageFont.ImageFont,
                  fill: tuple, line_height: int, max_y: int) -> int:
    """Draw word-wrapped text; return the y position after the last line."""
    for line in textwrap.wrap(text, width=width):
        if y >= max_y:
            draw.text((x, y), "…", fill=(160, 160, 160), font=font)
            return y
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def make_thumbnail(doc_code: str, doc: dict,
                   emails: list[dict] | None = None) -> str:
    """Render a mini document image; return as a base64 data URI."""
    color = CATEGORY_COLORS.get(doc["category"], DEFAULT_COLOR)
    color_rgb = _hex_rgb(color)

    img = Image.new("RGB", (THUMB_W, THUMB_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_sm  = _load_font(9)
    font_md  = _load_font(11)
    font_hdr = _load_font(13)

    # Coloured header bar
    header_h = 34
    draw.rectangle([(0, 0), (THUMB_W, header_h)], fill=color_rgb)
    draw.text((7, 5),  doc_code,          fill=(255, 255, 255), font=font_hdr)
    draw.text((7, 19), doc["category"],   fill=(210, 220, 230), font=font_sm)

    y = header_h + 6

    # Date line
    if doc.get("date"):
        draw.text((7, y), doc["date"].strftime("%-d %b %Y"),
                  fill=(80, 90, 100), font=font_sm)
        y += 14

    # Subject / title
    if emails:
        subj = (emails[0].get("subject") or "").strip()
        if not subj:
            subj = f"{len(emails)} e-mails"
    else:
        lines = [l.strip() for l in doc["text"].splitlines() if l.strip()]
        lines = [l for l in lines if not re.fullmatch(r"0\d{3}", l)]
        subj = " ".join(lines)[:70]

    y = _draw_wrapped(draw, subj, 7, y, width=29,
                      font=font_md, fill=(25, 30, 35),
                      line_height=14, max_y=y + 30)
    y += 5

    # Divider
    draw.line([(7, y), (THUMB_W - 7, y)], fill=(210, 215, 220), width=1)
    y += 7

    # Body preview
    body = (emails[0].get("text") or "") if emails else doc["text"]
    body = re.sub(r"\s+", " ", body).strip()
    _draw_wrapped(draw, body[:400], 7, y, width=33,
                  font=font_sm, fill=(65, 70, 80),
                  line_height=11, max_y=THUMB_H - 10)

    # Outer border
    draw.rectangle([(0, 0), (THUMB_W - 1, THUMB_H - 1)],
                   outline=(195, 200, 205), width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# HTML detail panel content
# ---------------------------------------------------------------------------
def _e(s) -> str:
    return html_mod.escape(str(s) if s is not None else "")


def _email_block_html(email: dict) -> str:
    fields = [("From", email.get("sender")),
              ("Date", email.get("date")),
              ("Subject", email.get("subject"))]
    hdr = "".join(
        f'<div class="em-row"><span class="em-lbl">{_e(k)}</span>{_e(v or "—")}</div>'
        for k, v in fields if v
    )
    body = _e(email.get("text", ""))
    subj = _e(email.get("subject") or "(geen onderwerp)")
    date = _e(email.get("date") or "")
    return (f'<details class="email-block" open>'
            f'<summary><span class="em-summary">{subj}</span>'
            f'<span class="em-date">{date}</span></summary>'
            f'<div class="em-header">{hdr}</div>'
            f'<pre class="em-body">{body}</pre></details>')


def _detail_html(doc_code: str, doc: dict, emails: list[dict] | None) -> str:
    date_str = doc["date"].strftime("%-d %B %Y") if doc["date"] else "Datum onbekend"
    cat_color = CATEGORY_COLORS.get(doc["category"], DEFAULT_COLOR)
    header = (f'<div class="det-header">'
              f'<span class="det-code">{_e(doc_code)}</span>'
              f'<span class="det-badge" style="background:{cat_color}">{_e(doc["category"])}</span>'
              f'<span class="det-date">{_e(date_str)}</span>'
              f'</div>')
    if emails:
        body = "\n".join(_email_block_html(e) for e in emails)
        return header + f'<div class="thread">{body}</div>'
    return header + f'<pre class="doc-pre">{_e(doc["text"])}</pre>'


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_html(docs: dict, out_path: Path | str = OUT_PATH) -> None:
    """Generate the timeline HTML file from an already-loaded, sorted docs dict."""

    # ── Build per-card data ──────────────────────────────────────────────────
    cards_data: list[dict] = []
    for doc_code, doc in docs.items():
        emails = (split_emails(doc["text"], doc_code)
                  if doc["category"] == "E-mail" else None)

        thumb   = make_thumbnail(doc_code, doc, emails)
        detail  = _detail_html(doc_code, doc, emails)
        color   = CATEGORY_COLORS.get(doc["category"], DEFAULT_COLOR)
        date_str = doc["date"].strftime("%-d %b %Y") if doc["date"] else "(geen datum)"

        if emails:
            title    = (emails[0].get("subject") or "").strip() or f"{len(emails)} e-mails"
            subtitle = f"{len(emails)} e-mail{'s' if len(emails) != 1 else ''} in thread"
        else:
            lines = [l.strip() for l in doc["text"].splitlines()
                     if l.strip() and not re.fullmatch(r"0\d{3}", l.strip())]
            title    = lines[0][:65] if lines else doc_code
            subtitle = f"{len(doc['pages'])} pagina{'s' if len(doc['pages']) != 1 else ''}"

        cards_data.append({
            "code":     doc_code,
            "category": doc["category"],
            "color":    color,
            "date":     date_str,
            "title":    title,
            "subtitle": subtitle,
            "thumb":    thumb,
            "detail":   detail,
        })

    # ── Cards HTML ───────────────────────────────────────────────────────────
    card_items = []
    for i, c in enumerate(cards_data):
        card_items.append(
            f'<div class="card" style="--cc:{_e(c["color"])}" onclick="openPanel({i})">'
            f'<div class="card-inner">'
            f'<div class="card-thumb">'
            f'<img src="{c["thumb"]}" alt="Doc {_e(c["code"])}" loading="lazy"/>'
            f'</div>'
            f'<div class="card-meta">'
            f'<span class="badge" style="background:{_e(c["color"])}">{_e(c["category"])}</span>'
            f'<div class="card-date">{_e(c["date"])}</div>'
            f'<div class="card-code">{_e(c["code"])}</div>'
            f'<div class="card-title" title="{_e(c["title"])}">{_e(c["title"])}</div>'
            f'<div class="card-sub">{_e(c["subtitle"])}</div>'
            f'</div></div></div>'
        )
    cards_html = "\n".join(card_items)

    panel_json = json.dumps(
        [{"code": c["code"], "title": c["title"], "date": c["date"],
          "detail": c["detail"]} for c in cards_data],
        ensure_ascii=False,
    )

    n = len(cards_data)

    # ── Assemble HTML ────────────────────────────────────────────────────────
    # Build JS and CSS separately to avoid f-string brace conflicts
    css = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;height:100vh;display:flex;
  flex-direction:column;overflow:hidden}
.hdr{padding:14px 24px 10px;background:#1e293b;border-bottom:1px solid #334155;flex-shrink:0}
.hdr h1{font-size:17px;font-weight:600;color:#f1f5f9;letter-spacing:.02em}
.hdr p{font-size:11px;color:#94a3b8;margin-top:2px}
.tl-outer{flex:1;overflow:hidden;display:flex;align-items:center}
.tl-scroll{overflow-x:auto;overflow-y:hidden;padding:52px 48px 20px;
  scrollbar-width:thin;scrollbar-color:#334155 #1e293b;width:100%}
.tl-scroll::-webkit-scrollbar{height:6px}
.tl-scroll::-webkit-scrollbar-track{background:#1e293b}
.tl-scroll::-webkit-scrollbar-thumb{background:#475569;border-radius:3px}
.tl-track{position:relative;display:flex;align-items:flex-start;
  min-width:max-content;gap:0}
.axis{position:absolute;top:-28px;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#334155 3%,#334155 97%,transparent)}
.card{position:relative;width:185px;flex-shrink:0;margin:0 10px;cursor:pointer;
  transition:transform .15s}
.card:hover{transform:translateY(-5px)}
.card:hover .card-inner{box-shadow:0 14px 36px rgba(0,0,0,.55)}
.card::before{content:'';position:absolute;top:-36px;left:50%;
  transform:translateX(-50%);width:11px;height:11px;border-radius:50%;
  background:var(--cc,#6b7280);border:2.5px solid #0f172a;z-index:1}
.card::after{content:'';position:absolute;top:-25px;left:50%;
  transform:translateX(-50%);width:1px;height:25px;
  background:var(--cc,#6b7280);opacity:.4}
.card-inner{background:#1e293b;border:1px solid #2d3f55;border-radius:8px;
  overflow:hidden;box-shadow:0 4px 18px rgba(0,0,0,.35);
  transition:box-shadow .15s,border-color .15s}
.card.active .card-inner{border-color:var(--cc,#6b7280);
  box-shadow:0 0 0 2px var(--cc,#6b7280),0 10px 28px rgba(0,0,0,.5)}
.card-thumb img{width:100%;display:block;border-bottom:1px solid #2d3f55}
.card-meta{padding:8px 9px 10px}
.badge{display:inline-block;font-size:9px;font-weight:700;color:#fff;
  padding:2px 6px;border-radius:3px;letter-spacing:.05em;text-transform:uppercase;
  margin-bottom:5px}
.card-date{font-size:11px;color:#94a3b8;margin-bottom:1px}
.card-code{font-size:10px;color:#475569;font-family:monospace;margin-bottom:3px}
.card-title{font-size:11px;color:#e2e8f0;font-weight:500;line-height:1.35;
  overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;
  -webkit-box-orient:vertical;margin-bottom:2px}
.card-sub{font-size:10px;color:#475569}
#panel{position:fixed;top:0;right:0;width:540px;height:100vh;
  background:#1e293b;border-left:1px solid #334155;display:flex;
  flex-direction:column;transform:translateX(100%);
  transition:transform .25s ease;z-index:100;
  box-shadow:-10px 0 40px rgba(0,0,0,.45)}
#panel.open{transform:translateX(0)}
.ptbar{display:flex;align-items:center;gap:10px;padding:13px 16px;
  border-bottom:1px solid #334155;flex-shrink:0}
.ptitle{flex:1;font-size:13px;font-weight:600;color:#f1f5f9;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#pclose{background:none;border:none;color:#64748b;cursor:pointer;
  font-size:22px;line-height:1;padding:1px 7px;border-radius:4px}
#pclose:hover{background:#334155;color:#f1f5f9}
#pbody{flex:1;overflow-y:auto;padding:16px;scrollbar-width:thin;
  scrollbar-color:#334155 #1e293b}
#pbody::-webkit-scrollbar{width:6px}
#pbody::-webkit-scrollbar-track{background:#1e293b}
#pbody::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
.det-header{display:flex;align-items:center;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.det-code{font-family:monospace;font-size:15px;font-weight:700;color:#f1f5f9}
.det-badge{font-size:9px;font-weight:700;color:#fff;padding:2px 8px;
  border-radius:3px;text-transform:uppercase;letter-spacing:.05em}
.det-date{font-size:12px;color:#94a3b8;margin-left:auto}
.thread{display:flex;flex-direction:column;gap:10px}
.email-block{background:#0f172a;border:1px solid #334155;border-radius:6px;overflow:hidden}
.email-block summary{display:flex;justify-content:space-between;align-items:center;
  padding:9px 12px;cursor:pointer;background:#162032;
  border-bottom:1px solid #334155;list-style:none;gap:8px;
  user-select:none}
.email-block summary::-webkit-details-marker{display:none}
.email-block:not([open]) summary{border-bottom:none}
.em-summary{font-size:12px;font-weight:600;color:#cbd5e1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.em-date{font-size:10px;color:#64748b;white-space:nowrap;flex-shrink:0}
.em-header{padding:8px 12px;background:#111c2d;border-bottom:1px solid #1e3050}
.em-row{font-size:11px;color:#94a3b8;margin-bottom:2px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.em-lbl{font-weight:600;color:#475569;display:inline-block;min-width:52px}
.em-body{font-family:'Courier New',monospace;font-size:11px;line-height:1.55;
  color:#cbd5e1;padding:12px;white-space:pre-wrap;word-break:break-word;
  max-height:320px;overflow-y:auto}
.doc-pre{font-family:'Courier New',monospace;font-size:11px;line-height:1.6;
  color:#cbd5e1;white-space:pre-wrap;word-break:break-word}
#ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:99}
#ov.on{display:block}
"""

    js = """
const DOCS=PANEL_JSON_PLACEHOLDER;
function openPanel(i){
  const d=DOCS[i];
  document.getElementById('ptitle').textContent=d.code+' \u2014 '+d.title;
  document.getElementById('pbody').innerHTML=d.detail;
  document.getElementById('panel').classList.add('open');
  document.getElementById('ov').classList.add('on');
  document.querySelectorAll('.card').forEach((c,j)=>c.classList.toggle('active',j===i));
}
function closePanel(){
  document.getElementById('panel').classList.remove('open');
  document.getElementById('ov').classList.remove('on');
  document.querySelectorAll('.card').forEach(c=>c.classList.remove('active'));
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closePanel();});
""".replace("PANEL_JSON_PLACEHOLDER", panel_json)

    page = (
        "<!DOCTYPE html>\n<html lang=\"nl\">\n<head>\n"
        "<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>WOO Document Timeline</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n<body>\n\n"
        "<div class=\"hdr\">\n"
        "  <h1>WOO Document Timeline</h1>\n"
        f"  <p>{n} documenten &mdash; klik op een kaart voor de volledige tekst</p>\n"
        "</div>\n\n"
        "<div class=\"tl-outer\">\n"
        "  <div class=\"tl-scroll\">\n"
        "    <div class=\"tl-track\">\n"
        "      <div class=\"axis\"></div>\n"
        f"      {cards_html}\n"
        "    </div>\n"
        "  </div>\n"
        "</div>\n\n"
        "<div id=\"ov\" onclick=\"closePanel()\"></div>\n\n"
        "<div id=\"panel\">\n"
        "  <div class=\"ptbar\">\n"
        "    <span class=\"ptitle\" id=\"ptitle\"></span>\n"
        "    <button id=\"pclose\" onclick=\"closePanel()\" title=\"Sluiten\">&times;</button>\n"
        "  </div>\n"
        "  <div id=\"pbody\"></div>\n"
        "</div>\n\n"
        f"<script>{js}</script>\n"
        "</body>\n</html>"
    )

    Path(out_path).write_text(page, encoding="utf-8")
    size_kb = Path(out_path).stat().st_size // 1024
    print(f"Written: {out_path}  ({size_kb} KB, {n} cards)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pdf_import_reader import load_pdf
    from text_sorting import sort_documents
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_PATH
    build_html(sort_documents(load_pdf(Path(pdf))), out)
