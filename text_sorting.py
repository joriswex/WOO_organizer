"""
text_sorting.py — date extraction and chronological sorting of WOO documents.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pdf_import_reader import load_pdf
from email_splitter import split_emails

# ---------------------------------------------------------------------------
# Month name tables (Dutch + English)
# ---------------------------------------------------------------------------
_MONTHS: dict[str, int] = {
    # Dutch
    'januari': 1, 'februari': 2, 'maart': 3, 'april': 4,
    'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
    'september': 9, 'oktober': 10, 'november': 11, 'december': 12,
    # English (non-overlapping; april/september/november/december are identical)
    'january': 1, 'february': 2, 'march': 3,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'october': 10,
}

# Longest-first so regex alternation never partially matches a shorter name
_MONTH_PAT = '|'.join(sorted(_MONTHS, key=len, reverse=True))


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------
def _parse_date(raw: str) -> datetime | None:
    """Parse a date string in common WOO formats. Returns None if unrecognised."""
    if not raw:
        return None
    s = raw.strip()

    # "Mon 3/17/2025 7:23:28 PM" / "Thur 6/5/2025 12:58:26 PM"  (short weekday + M/D/YYYY)
    m = re.match(r'\w{2,5}\.?\s+(\d{1,2})/(\d{1,2})/(\d{4})', s)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    # "Wednesday, March 25, 2025 10:42:47 AM"  (English long weekday)
    m = re.match(r'\w+,\s+(\w+)\s+(\d{1,2}),\s+(\d{4})', s)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(2)))
            except ValueError:
                pass

    # "maandag 17 maart 2025 10:02"  (Dutch/English weekday + DD month YYYY)
    m = re.match(r'\w+\s+(\d{1,2})\s+(\w+)\s+(\d{4})', s)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                pass

    # "18-09-2025"  (DD-MM-YYYY, exact match)
    m = re.fullmatch(r'(\d{1,2})-(\d{1,2})-(\d{4})', s)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # "9december2024" / "17maart2025" / "nota 25februari 2025"  (glued or space-separated)
    m = re.search(rf'(\d{{1,2}})\s*({_MONTH_PAT})\s*(\d{{4}})', s, re.IGNORECASE)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                pass

    return None


# ---------------------------------------------------------------------------
# Category-specific extractors
# ---------------------------------------------------------------------------
def _date_from_email(doc: dict) -> tuple[datetime | None, str]:
    """Sent date of the first email in the thread that has a parseable date."""
    emails = split_emails(doc['text'], doc['doc_code'])
    for email in emails:
        raw = (email.get('date') or '').strip()
        if raw:
            dt = _parse_date(raw)
            if dt:
                return dt, raw
    return None, 'no parseable sent-date in email headers'


def _date_from_datum_field(text: str) -> tuple[datetime | None, str]:
    """Extract the Datum: / Datum | field from the first 600 chars."""
    m = re.search(r'(?i)\bdatum\b\s*[|\n:]\s*(.{2,50}?)(?:\s*\||\n|$)', text[:600])
    if m:
        raw = m.group(1).strip()
        dt = _parse_date(raw)
        if dt:
            return dt, raw
        return None, f'Datum field found but unparseable: {raw!r}'
    return None, 'no Datum field in header'


def _date_from_text_scan(text: str, chars: int = 1000) -> tuple[datetime | None, str]:
    """Scan the first `chars` characters for any recognisable date pattern."""
    snippet = text[:chars]

    # DD-MM-YYYY
    for m in re.finditer(r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', snippet):
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))), m.group(0)
        except ValueError:
            pass

    # DD monthname YYYY  (glued or spaced, NL/EN)
    for m in re.finditer(rf'(\d{{1,2}})\s*({_MONTH_PAT})\s*(\d{{4}})', snippet, re.IGNORECASE):
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(1))), m.group(0)
            except ValueError:
                pass

    return None, 'no date found in document text'


def _get_doc_date(doc: dict) -> tuple[datetime | None, str]:
    category = doc['category']

    if category == 'E-mail':
        return _date_from_email(doc)

    if category in ('Nota', 'Brief', 'Report', 'Vergadernotulen'):
        dt, raw = _date_from_datum_field(doc['text'])
        if dt:
            return dt, raw
        return _date_from_text_scan(doc['text'])

    # Timeline, Other, and any unknown category
    return _date_from_text_scan(doc['text'])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def sort_documents(docs: dict) -> dict:
    """
    Attach ``date`` (datetime | None) and ``date_raw`` (str) to every document
    and return them sorted chronologically.
    Undated documents are appended at the end, sorted by doc_code.
    """
    for doc in docs.values():
        dt, raw = _get_doc_date(doc)
        doc['date'] = dt
        doc['date_raw'] = raw

    dated = sorted(
        [(c, d) for c, d in docs.items() if d['date'] is not None],
        key=lambda x: x[1]['date'],
    )
    undated = sorted(
        [(c, d) for c, d in docs.items() if d['date'] is None],
        key=lambda x: x[0],
    )

    return {code: doc for code, doc in dated + undated}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else 'test.pdf'
    docs = sort_documents(load_pdf(Path(pdf)))

    print(f"\n{'#':<4} {'Code':<8} {'Date':<14} {'Category':<20} {'Date source'}")
    print('-' * 78)
    for rank, (code, doc) in enumerate(docs.items(), 1):
        dt_str = doc['date'].strftime('%d-%m-%Y') if doc['date'] else '(no date)'
        flag = '  ← WARNING' if doc['date'] is None else ''
        print(f"{rank:<4} {code:<8} {dt_str:<14} {doc['category']:<20} "
              f"{doc['date_raw'][:35]}{flag}")
