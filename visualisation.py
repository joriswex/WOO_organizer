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
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from email_splitter import split_emails

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_PATH = Path("woo_timeline.html")

THUMB_W, THUMB_H = 200, 280

CATEGORY_COLORS = {
    "E-mail":          "#2563EB",
    "Chat":            "#059669",
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
    """Load a system font at *size* pt, falling back to Pillow's built-in default."""
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
    """Convert a ``#RRGGBB`` hex string to an ``(R, G, B)`` integer tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                  width: int, font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
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
                   emails: list[dict] | None = None,
                   page_image: Image.Image | None = None) -> str:
    """Render a mini document image; return as a base64 data URI.

    For non-email documents a rendered PDF page (page_image) is used when
    available so the original formatting is preserved.  Email documents always
    use the text-based render so the subject/body preview is shown.
    """
    # ── PDF-page thumbnail (non-email docs) ──────────────────────────────────
    if page_image is not None and emails is None:
        canvas = Image.new("RGB", (THUMB_W, THUMB_H), (248, 248, 248))
        page = page_image.copy()
        page.thumbnail((THUMB_W - 4, THUMB_H - 4), Image.Resampling.LANCZOS)
        x = (THUMB_W - page.width) // 2
        y = (THUMB_H - page.height) // 2
        canvas.paste(page, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (THUMB_W - 1, THUMB_H - 1)],
                       outline=(195, 200, 205), width=1)
        buf = io.BytesIO()
        canvas.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ── Text-based thumbnail (email docs or fallback) ────────────────────────
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
    draw.text((7, 5),  doc_code,         fill=(255, 255, 255), font=font_hdr)
    draw.text((7, 19), doc["category"],  fill=(210, 220, 230), font=font_sm)

    y = header_h + 6

    # Date line
    if doc.get("date"):
        draw.text((7, y), doc["date"].strftime("%-d %b %Y"),
                  fill=(80, 90, 100), font=font_sm)
        y += 14

    # Subject / title
    if emails:
        subj = (emails[0].get("subject") or "").strip() or f"{len(emails)} e-mails"
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

    # Body preview (emails: strip headers from preview text too)
    if emails:
        body = _email_body(emails[0].get("text") or "")
    else:
        body = doc["text"]
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
# Ministry domain → full name lookup
# ---------------------------------------------------------------------------
_MINISTRY_DOMAINS: dict[str, str] = {
    # Core ministries
    "minbuza.nl":       "Ministerie van Buitenlandse Zaken",
    "minjenv.nl":       "Ministerie van Justitie en Veiligheid",
    "minbzk.nl":        "Ministerie van Binnenlandse Zaken",
    "minfin.nl":        "Ministerie van Financiën",
    "minocw.nl":        "Ministerie van Onderwijs, Cultuur en Wetenschap",
    "minvws.nl":        "Ministerie van Volksgezondheid, Welzijn en Sport",
    "minlnv.nl":        "Ministerie van Landbouw, Natuur en Voedselkwaliteit",
    "minszw.nl":        "Ministerie van Sociale Zaken en Werkgelegenheid",
    "minienw.nl":       "Ministerie van Infrastructuur en Waterstaat",
    "minav.nl":         "Ministerie van Algemene Zaken",
    "minez.nl":         "Ministerie van Economische Zaken",
    "minezk.nl":        "Ministerie van Economische Zaken en Klimaat",
    "minelzk.nl":       "Ministerie van Economische Zaken en Klimaat",
    "defensie.nl":      "Ministerie van Defensie",
    # Agencies & other government
    "rijksoverheid.nl": "Rijksoverheid",
    "nctv.nl":          "NCTV",
    "ind.nl":           "IND",
    "rvo.nl":           "RVO",
    "coa.nl":           "COA",
    "rijksrecherche.nl":"Rijksrecherche",
    "aivd.nl":          "AIVD",
    "mivd.nl":          "MIVD",
    "politie.nl":       "Politie",
    "om.nl":            "Openbaar Ministerie",
    "rechtspraak.nl":   "Rechtspraak",
    "tweedekamer.nl":   "Tweede Kamer",
    "eerstekamer.nl":   "Eerste Kamer",
    "raadvanstate.nl":  "Raad van State",
    "kabinetsformatie.nl": "Kabinetsformatie",
    "parlement.com":    "Parlement",
}

# Regex to extract @domain from any address string (including [REDACTED]@domain.nl)
_DOMAIN_RE = re.compile(r"@([\w.-]+\.\w{2,})", re.IGNORECASE)
# WOO redaction code for per-email counting
_EMAIL_REDACT_RE = re.compile(r"5\.[12]\.[1-9][a-z]{0,2}", re.IGNORECASE)


def _classify_domain(domain: str) -> tuple[str, str]:
    """Return (display_label, tag_type) for an email domain.

    tag_type is one of: 'ministry', 'govt', 'external'
    """
    d = domain.lower()
    ministry = _MINISTRY_DOMAINS.get(d)
    if ministry:
        tag_type = "ministry"
        return ministry, tag_type
    # Abbreviate long ministry names to keep tags readable
    return f"@{domain} (Extern)", "external"


def _parse_recipients(addr_str: str) -> list[tuple[str, str]]:
    """Split a recipient string into individual (label, tag_type) tuples.

    Handles semicolon/comma-separated lists and [REDACTED]@domain.nl format.
    """
    if not addr_str:
        return []

    # Split on ; or ,  but not inside angle brackets
    parts = re.split(r"[;,](?![^<]*>)", addr_str)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for part in parts:
        part = part.strip()
        if not part:
            continue

        domains = _DOMAIN_RE.findall(part)
        if domains:
            for domain in domains:
                label, tag_type = _classify_domain(domain)
                key = label.lower()
                if key not in seen:
                    seen.add(key)
                    results.append((label, tag_type))
        else:
            # Fully redacted — no domain visible
            label = "[Redacted]"
            if label.lower() not in seen:
                seen.add(label.lower())
                results.append((label, "unknown"))

    return results


def _recipient_tags_html(addr_str: str) -> str:
    """Render recipient address string as individual pill tags."""
    recipients = _parse_recipients(addr_str)
    if not recipients:
        return _e(addr_str) if addr_str else ""
    return " ".join(
        f'<span class="em-rcpt em-rcpt-{tag_type}">{_e(label)}</span>'
        for label, tag_type in recipients
    )


def _recipients_row_html(field_label: str, addr_str: str) -> str:
    """Render a header row (Van/Aan/CC) with individual recipient tags."""
    if not addr_str:
        return ""
    tags = _recipient_tags_html(addr_str)
    return (
        f'<div class="em-field em-field-recipients">'
        f'<span class="em-label">{_e(field_label)}</span>'
        f'<span class="em-val em-rcpts">{tags}</span>'
        f'</div>'
    )


def _email_redact_summary_html(text: str) -> str:
    """Build a compact redaction-code summary for a single email."""
    counts: Counter[str] = Counter(
        m.lower() for m in _EMAIL_REDACT_RE.findall(text)
    )
    if not counts:
        return ""
    total = sum(counts.values())
    items = "".join(
        f'<span class="em-redact-item">'
        f'<span class="em-redact-code">{_e(code)}</span>'
        f'<span class="em-redact-n">{n}\u00d7</span>'
        f'<span class="em-redact-pct">{n/total*100:.0f}%</span>'
        f'</span>'
        for code, n in sorted(counts.items(), key=lambda x: -x[1])
    )
    return (
        f'<div class="em-redact-summary">'
        f'<span class="em-redact-label">Redacties ({total})</span>'
        f'{items}'
        f'</div>'
    )


def _resolve_sender(raw: str | None) -> str:
    """Return a human-readable sender label, with ministry names where known.

    Input may be 'Name <email@domain.nl>' or just 'email@domain.nl'.
    When the domain matches a known ministry the ministry name is appended.
    """
    if not raw:
        return "Onbekend"
    # Extract display name and email
    m = re.match(r'^([^<\n]+?)\s*<([^>]+)>', raw.strip())
    if m:
        name, email = m.group(1).strip(), m.group(2).strip()
    else:
        name, email = "", raw.strip()

    domain = email.split("@")[-1].lower() if "@" in email else ""
    ministry = _MINISTRY_DOMAINS.get(domain, "")

    if name and ministry:
        return f"{name} ({ministry})"
    if ministry:
        return ministry
    if name:
        return name
    return email or raw


def _resolve_addr_in_text(text: str) -> str:
    """Replace ministry email addresses in body text with human-readable names.

    'jan.de.vries@minbuza.nl' → 'Jan De Vries (Ministerie van Buitenlandse Zaken)'
    Unknown domains are left unchanged.
    """
    def _repl(m: re.Match) -> str:
        local    = m.group(1)
        domain   = m.group(2).lower()
        ministry = _MINISTRY_DOMAINS.get(domain)
        if not ministry:
            return m.group(0)
        name = local.replace(".", " ").replace("_", " ").title()
        return f"{name} ({ministry})"
    return re.sub(r"\b([\w.+-]+)@([\w.-]+\.\w+)", _repl, text)


# ---------------------------------------------------------------------------
# Email body boilerplate stripper
# ---------------------------------------------------------------------------
_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"denk\s+aan\s+het\s+milieu"
    r"|please\s+consider\s+the\s+environment"
    r"|help\s+save\s+paper"
    r"|think\s+before\s+you\s+print"
    r"|think\s+before\s+printing"
    r"|overweeg\s+de\s+natuur"
    r"|dit\s+(?:e-?mail)?bericht\s+(?:en\s+bijlagen\s+)?(?:is\s+)?(?:uitsluitend\s+bestemd|vertrouwelijk)"
    r"|this\s+(?:e-?mail|message)\s+(?:and\s+(?:any\s+)?attachments?\s+)?(?:is\s+)?(?:confidential|intended\s+only\s+for)"
    r"|de\s+informatie\s+in\s+dit\s+(?:e-?mail)?bericht"
    r")",
    re.IGNORECASE,
)


def _strip_email_boilerplate(text: str) -> str:
    """Remove common footer boilerplate (save-paper notices, confidentiality
    disclaimers) that appear after the main body text.

    Scans line by line; once a boilerplate trigger is found, everything from
    that line onward is dropped, then trailing blank lines are removed.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _BOILERPLATE_RE.search(line):
            lines = lines[:i]
            break
    # Trim trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Redaction code display helper
# ---------------------------------------------------------------------------
_BODY_REDACT_RE = re.compile(r"\b5\.[12]\.[1-9][a-z]{0,2}\b", re.IGNORECASE)


def _redact_body_html(text: str) -> str:
    """Replace redaction codes in body text with a styled (REDACTED) span."""
    escaped = html_mod.escape(text)
    return _BODY_REDACT_RE.sub(
        '<span class="redacted">(REDACTED)</span>', escaped
    )


# ---------------------------------------------------------------------------
# Email body stripper
# ---------------------------------------------------------------------------
_HEADER_FIELD_RE = re.compile(
    r"^[ \t]*(?:from|van|to|aan|subject|onderwerp|sent|verzonden|received|cc|bcc)\s*:",
    re.IGNORECASE,
)


def _email_body(text: str, scan_lines: int = 30) -> str:
    """Return *text* with the leading email header block removed.

    Scans the first *scan_lines* lines for recognised header field lines
    (From/Van, To/Aan, Subject/Onderwerp, Sent/Verzonden, Received, CC, BCC).
    Everything up to and including the last header field (plus any indented
    continuation lines) is stripped; leading blank lines after the block are
    also removed.
    """
    lines = text.splitlines()
    last_header = -1
    for i, line in enumerate(lines[:scan_lines]):
        if _HEADER_FIELD_RE.match(line):
            last_header = i

    if last_header == -1:
        return text  # no header block found — return as-is

    # Advance past any continuation lines (indented) right after last header
    i = last_header + 1
    while i < len(lines) and i < scan_lines and lines[i].startswith((" ", "\t")):
        i += 1
    # Skip leading blank lines before the body
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


# ---------------------------------------------------------------------------
# HTML detail panel content
# ---------------------------------------------------------------------------
def _e(s) -> str:
    """HTML-escape *s*, converting None to an empty string."""
    return html_mod.escape(str(s) if s is not None else "")


def _redaction_summary_html(code_counts: dict[str, int]) -> str:
    """Build a compact per-code count/percentage block for the detail panel.

    Codes with a letter suffix (e.g. 5.1.2e) are highlighted in amber —
    they indicate a specific sub-ground and are of special interest.
    """
    if not code_counts:
        return ""

    total = sum(code_counts.values())

    rows: list[str] = []
    for code, n in sorted(code_counts.items(), key=lambda x: -x[1]):
        pct      = n / total * 100
        bar_w    = max(2, round(pct))   # at least 2 % wide so hairline is visible
        suffix   = re.search(r"[a-z]$", code)
        if suffix:
            code_cls = "redact-code has-letter"
            bar_cls  = "redact-bar has-letter"
        else:
            code_cls = "redact-code"
            bar_cls  = "redact-bar"
        label = _e(code)
        rows.append(
            f'<div class="redact-row">'
            f'<span class="{code_cls}">{label}</span>'
            f'<div class="redact-bar-wrap">'
            f'<div class="{bar_cls}" style="width:{bar_w}%"></div></div>'
            f'<span class="redact-pct">{pct:.0f}%</span>'
            f'<span class="redact-count">{n}\u00d7</span>'
            f'</div>'
        )

    return (
        f'<div class="redact-section">'
        f'<div class="redact-title">Redactieoverzicht'
        f'<span class="redact-total">{total} passage{"s" if total != 1 else ""}</span>'
        f'</div>'
        f'<div class="redact-rows">{"".join(rows)}</div>'
        f'</div>'
    )


def _chat_thread_html(chat_name: str | None, messages: list[dict]) -> str:
    """Render a messaging-app conversation as chat bubbles.

    Left bubbles = incoming (other participants).
    Right bubbles = outgoing (device owner).
    Senders are typically redacted; we use the sender_label as an identifier
    and assign a consistent colour to each unique label.
    """
    if not messages:
        return '<div class="chat-empty">Geen berichten gevonden</div>'

    # Assign a colour index to each unique sender_label (left side only)
    _LEFT_COLORS = ["#1d4ed8", "#0f766e", "#7c3aed", "#b45309", "#be185d", "#0369a1"]
    sender_colors: dict[str, str] = {}
    color_idx = 0
    for m in messages:
        lbl = (m.get("sender_label") or "").strip()
        if m.get("sender_position") != "right" and lbl and lbl not in sender_colors:
            sender_colors[lbl] = _LEFT_COLORS[color_idx % len(_LEFT_COLORS)]
            color_idx += 1

    header = ""
    if chat_name:
        header = f'<div class="chat-name">{_e(chat_name)}</div>'

    bubbles: list[str] = []
    for m in messages:
        pos       = m.get("sender_position") or "left"
        label     = (m.get("sender_label") or "").strip()
        ts        = (m.get("timestamp") or "").strip()
        content   = (m.get("content") or "").strip()

        # Wrap redaction codes in styled spans
        content_html = re.sub(
            r"\b(5\.[12]\.[1-9][a-z]{0,2})\b",
            r'<span class="chat-redact">\1</span>',
            _e(content),
            flags=re.IGNORECASE,
        )
        content_html = content_html.replace("[REDACTED]", '<span class="chat-redact">[REDACTED]</span>')

        ts_html    = f'<span class="chat-ts">{_e(ts)}</span>' if ts else ""
        side_cls   = "chat-bubble-right" if pos == "right" else "chat-bubble-left"
        row_cls    = "chat-row-right" if pos == "right" else "chat-row-left"

        if pos == "right":
            label_html = ""
        else:
            color      = sender_colors.get(label, "#475569")
            label_html = f'<div class="chat-sender" style="color:{color}">{_e(label) or "Afzender"}</div>'

        bubbles.append(
            f'<div class="{row_cls}">'
            f'<div class="{side_cls}">'
            f'{label_html}'
            f'<div class="chat-content">{content_html}</div>'
            f'{ts_html}'
            f'</div>'
            f'</div>'
        )

    return (
        f'<div class="chat-thread">'
        f'{header}'
        f'<div class="chat-messages">{"".join(bubbles)}</div>'
        f'</div>'
    )


def _detail_html(doc_code: str, doc: dict, emails: list[dict] | None) -> str:
    """Build the HTML string for the side-panel detail view of one document."""
    date_str = doc["date"].strftime("%-d %B %Y") if doc["date"] else "Datum onbekend"
    cat_color = CATEGORY_COLORS.get(doc["category"], DEFAULT_COLOR)
    header = (f'<div class="det-header">'
              f'<span class="det-code">{_e(doc_code)}</span>'
              f'<span class="det-badge" style="background:{cat_color}">{_e(doc["category"])}</span>'
              f'<span class="det-date">{_e(date_str)}</span>'
              f'</div>')
    redact = _redaction_summary_html(doc.get("redaction_codes") or {})
    if doc.get("category") == "Chat" and doc.get("chat_messages"):
        return header + redact + _chat_thread_html(doc.get("chat_name"), doc["chat_messages"])
    if emails:
        return header + redact + _email_thread_html(emails)
    return header + redact  # PDF page images are injected by JS via PAGE_IMGS


# Attachment detection --------------------------------------------------------

# Matches "Bijlage(n): ..." or "Attachment(s): ..." header lines
_ATTACH_HEADER_RE = re.compile(
    r"^(?:bijlagen?|meegezonden|meegestuurde?\s+(?:document|bestand)en?|attachments?)\s*:?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
# Matches a line that is (optionally bullet-prefixed) just a filename
_ATTACH_FILENAME_LINE_RE = re.compile(
    r"^[-•*·]?\s*([\w\-. ()\[\]]+\.(?:pdf|docx?|xlsx?|pptx?|txt|zip|csv|msg|eml))\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Any filename-looking token anywhere in a line (used for header-line parsing)
_FILENAME_TOKEN_RE = re.compile(
    r"[\w\-. ()\[\]]+\.(?:pdf|docx?|xlsx?|pptx?|txt|zip|csv|msg|eml)",
    re.IGNORECASE,
)


def _strip_attachment_lines(text: str) -> str:
    """Remove attachment header lines and standalone filename lines from body text."""
    text = _ATTACH_HEADER_RE.sub("", text)
    text = _ATTACH_FILENAME_LINE_RE.sub("", text)
    # Collapse runs of blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_attachments(text: str) -> list[str]:
    """Return a deduplicated list of attachment filenames found in the email text."""
    found: list[str] = []
    seen:  set[str]  = set()

    def _add(name: str) -> None:
        key = name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            found.append(name.strip())

    # 1. Explicit attachment header lines: "Bijlage: rapport.pdf, brief.pdf"
    for m in _ATTACH_HEADER_RE.finditer(text):
        remainder = m.group(1).strip()
        for fn in _FILENAME_TOKEN_RE.findall(remainder):
            _add(fn)

    # 2. Lines that consist of nothing but a filename
    for m in _ATTACH_FILENAME_LINE_RE.finditer(text):
        _add(m.group(1))

    return found


def _attachments_html(attachments: list[str]) -> str:
    if not attachments:
        return ""
    tags = "".join(
        f'<span class="em-attach-tag">📎 {_e(a)}</span>'
        for a in attachments
    )
    return f'<div class="em-attachments">{tags}</div>'


def _email_thread_html(emails: list[dict]) -> str:
    """Render each email as a structured block with header fields and body text."""
    blocks: list[str] = []
    for email in emails:
        eid     = _e(email.get("id") or "")
        sender  = _e(_resolve_sender(email.get("sender")))
        to_val  = _e(_resolve_sender(email.get("to") or "")) if email.get("to") else ""
        cc_val  = _e(_resolve_sender(email.get("cc") or "")) if email.get("cc") else ""
        subject = _e(email.get("subject") or "(geen onderwerp)")
        date    = _e(email.get("date") or "")

        body_raw    = _email_body(email.get("text") or "")
        body_raw    = _strip_email_boilerplate(body_raw)
        body_raw    = _resolve_addr_in_text(body_raw)
        attachments = _extract_attachments(body_raw)
        if attachments:
            body_raw = _strip_attachment_lines(body_raw)
        body_html   = _redact_body_html(body_raw)

        warning     = email.get("warning")
        warn_html   = (f'<div class="em-warn">⚠ {_e(warning)}</div>' if warning else "")
        attach_html = _attachments_html(attachments)
        redact_html = _email_redact_summary_html(email.get("text") or "")

        van_row = _recipients_row_html("Van",  email.get("sender") or "")
        aan_row = _recipients_row_html("Aan",  email.get("to")     or "")
        cc_row  = _recipients_row_html("CC",   email.get("cc")     or "")

        blocks.append(
            f'<div class="em-block">'
            f'<div class="em-head">'
            f'<div class="em-id">{eid}</div>'
            f'{van_row}'
            f'{aan_row}'
            f'{cc_row}'
            f'<div class="em-field"><span class="em-label">Onderwerp</span><span class="em-val em-subj">{subject}</span></div>'
            f'<div class="em-field"><span class="em-label">Datum</span><span class="em-val em-date">{date}</span></div>'
            f'</div>'
            f'{redact_html}'
            f'{attach_html}'
            f'{warn_html}'
            f'<div class="em-body">{body_html}</div>'
            f'</div>'
        )
    return f'<div class="em-thread">{"".join(blocks)}</div>'


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_html(docs: dict, out_path: Path | str = OUT_PATH,
               pdf_path: Path | str | None = None) -> None:
    """Generate the timeline HTML file from an already-loaded, sorted docs dict."""

    # ── Render PDF pages: thumbnails (first page) + panel images (all pages) ──
    page_images: dict[str, Image.Image] = {}   # first-page PIL for card thumbnail
    doc_page_uris: dict[str, list[str]] = {}   # all pages as JPEG base64 for panel
    if pdf_path is not None and Path(pdf_path).exists():
        try:
            import pdfplumber
            all_docs = [(c, d) for c, d in docs.items() if d.get("pages")]
            with pdfplumber.open(pdf_path) as pdf:
                for doc_code, doc in all_docs:
                    print(f"  [{doc_code}] rendering {len(doc['pages'])} page(s)…",
                          flush=True)
                    uris: list[str] = []
                    for j, page_num in enumerate(doc["pages"]):
                        try:
                            img = pdf.pages[page_num - 1].to_image(resolution=100).original
                            if j == 0:
                                page_images[doc_code] = img   # thumbnail source
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=65, optimize=True)
                            uris.append(
                                "data:image/jpeg;base64,"
                                + base64.b64encode(buf.getvalue()).decode()
                            )
                        except Exception:
                            pass
                    if uris:
                        doc_page_uris[doc_code] = uris
        except Exception:
            pass

    # ── Build per-card data ──────────────────────────────────────────────────
    cards_data: list[dict] = []
    for doc_code, doc in docs.items():
        emails = (split_emails(doc["text"], doc_code)
                  if doc["category"] == "E-mail" else None)

        thumb   = make_thumbnail(doc_code, doc, emails, page_images.get(doc_code))
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
            "code":        doc_code,
            "category":    doc["category"],
            "color":       color,
            "date":        date_str,
            "title":       title,
            "subtitle":    subtitle,
            "thumb":       thumb,
            "detail":      detail,
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
.doc-pre{font-family:'Courier New',monospace;font-size:11px;line-height:1.6;
  color:#cbd5e1;white-space:pre-wrap;word-break:break-word}
.pdf-pages{display:flex;flex-direction:column;gap:6px;margin-top:12px}
.pdf-page{width:100%;display:block;border:1px solid #2d3f55;border-radius:2px}
.redact-section{background:#111c2d;border:1px solid #1e3050;border-radius:6px;
  padding:10px 12px;margin-bottom:14px}
.redact-title{font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:7px;
  display:flex;align-items:center;justify-content:space-between}
.redact-total{font-weight:400;color:#475569;font-size:10px}
.redact-row{display:grid;grid-template-columns:72px 1fr 36px 32px;
  align-items:center;gap:7px;margin-bottom:4px}
.redact-code{font-family:monospace;font-size:11px;font-weight:600;color:#64748b}
.redact-code.has-letter{color:#fbbf24}
.redact-bar-wrap{height:5px;background:#1e2d3d;border-radius:3px;overflow:hidden}
.redact-bar{height:100%;border-radius:3px;background:#334155}
.redact-bar.has-letter{background:#d97706}
.redact-pct{font-size:10px;color:#64748b;text-align:right}
.redact-count{font-size:10px;color:#475569;text-align:right}
.em-thread{display:flex;flex-direction:column;gap:10px;margin-top:8px}
.em-block{background:#0f1e30;border:1px solid #1e3050;border-radius:6px;overflow:hidden}
.em-head{padding:10px 12px;border-bottom:1px solid #1e3050;background:#111c2d}
.em-id{font-family:monospace;font-size:9px;color:#475569;margin-bottom:6px}
.em-field{display:flex;gap:8px;margin-bottom:3px;font-size:11px;line-height:1.4}
.em-label{color:#64748b;font-weight:600;min-width:72px;flex-shrink:0}
.em-val{color:#cbd5e1;word-break:break-word}
.em-from{color:#93c5fd}
.em-to{color:#6ee7b7}
.em-subj{color:#f1f5f9;font-weight:600}
.em-date{color:#94a3b8}
.em-field-recipients{align-items:flex-start}
.em-rcpts{display:flex;flex-wrap:wrap;gap:4px}
.em-rcpt{display:inline-flex;align-items:center;padding:2px 7px;border-radius:10px;
  font-size:10px;font-weight:500;white-space:nowrap;line-height:1.4}
.em-rcpt-ministry{background:#1a3a5c;border:1px solid #2d5a8c;color:#93c5fd}
.em-rcpt-external{background:#2d2010;border:1px solid #5a4010;color:#fbbf24}
.em-rcpt-unknown{background:#1e1e2e;border:1px solid #3a3a5c;color:#64748b}
.em-redact-summary{padding:5px 12px;display:flex;flex-wrap:wrap;align-items:center;
  gap:8px;border-bottom:1px solid #1e3050;background:#0d1520}
.em-redact-label{font-size:9px;color:#475569;font-weight:600;text-transform:uppercase;
  letter-spacing:.04em;margin-right:4px}
.em-redact-item{display:inline-flex;align-items:center;gap:3px;
  background:#1a1a2e;border:1px solid #2d2d4e;border-radius:8px;padding:1px 6px}
.em-redact-code{font-size:9px;color:#c084fc;font-family:monospace}
.em-redact-n{font-size:9px;color:#94a3b8}
.em-redact-pct{font-size:9px;color:#475569}
.em-attachments{padding:6px 12px;display:flex;flex-wrap:wrap;gap:5px;
  border-bottom:1px solid #1e3050;background:#0d1a2b}
.em-attach-tag{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;
  background:#1e3050;border:1px solid #2d4a70;border-radius:12px;
  font-size:10px;color:#93c5fd;white-space:nowrap}
.chat-thread{display:flex;flex-direction:column;gap:0;margin-top:8px}
.chat-name{text-align:center;font-size:10px;font-weight:600;color:#64748b;
  padding:6px 12px;background:#0a1220;border-bottom:1px solid #1e3050;letter-spacing:.04em}
.chat-messages{display:flex;flex-direction:column;gap:4px;padding:10px 10px}
.chat-row-left{display:flex;justify-content:flex-start}
.chat-row-right{display:flex;justify-content:flex-end}
.chat-bubble-left,.chat-bubble-right{max-width:78%;padding:6px 10px;border-radius:12px;
  font-size:11px;line-height:1.55;word-break:break-word}
.chat-bubble-left{background:#1e3050;border-top-left-radius:3px}
.chat-bubble-right{background:#0f3d2e;border-top-right-radius:3px}
.chat-sender{font-size:9px;font-weight:700;margin-bottom:2px}
.chat-content{color:#cbd5e1;white-space:pre-wrap}
.chat-ts{display:block;font-size:9px;color:#475569;text-align:right;margin-top:2px}
.chat-redact{background:#3b1f4e;color:#c084fc;border-radius:3px;
  padding:0 3px;font-size:9px;font-family:monospace}
.chat-empty{padding:12px;font-size:11px;color:#475569;text-align:center}
.em-warn{background:#1f1207;color:#f59e0b;font-size:10px;padding:5px 12px;
  border-bottom:1px solid #2d1a00}
.em-body{padding:10px 12px;font-size:11px;line-height:1.65;color:#94a3b8;
  white-space:pre-wrap;word-break:break-word;max-height:280px;overflow-y:auto}
#ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:99}
#ov.on{display:block}
"""

    page_imgs_json = json.dumps(doc_page_uris, ensure_ascii=False)

    js = """
const DOCS=PANEL_JSON_PLACEHOLDER;
const PAGE_IMGS=PAGE_IMGS_JSON_PLACEHOLDER;
function openPanel(i){
  const d=DOCS[i];
  document.getElementById('ptitle').textContent=d.code+' \u2014 '+d.title;
  const body=document.getElementById('pbody');
  body.innerHTML=d.detail;
  const imgs=PAGE_IMGS[d.code];
  if(imgs){
    const wrap=document.createElement('div');
    wrap.className='pdf-pages';
    imgs.forEach(src=>{
      const im=new Image();
      im.src=src;
      im.className='pdf-page';
      wrap.appendChild(im);
    });
    body.appendChild(wrap);
  }
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
""".replace("PANEL_JSON_PLACEHOLDER", panel_json
   ).replace("PAGE_IMGS_JSON_PLACEHOLDER", page_imgs_json)

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
