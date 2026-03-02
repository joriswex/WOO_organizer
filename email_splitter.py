"""
e-mail_splitter.py

Splits every E-mail document returned by load_pdf() into individual emails.

Both Dutch Outlook and English Outlook formats are supported:

  Dutch  : Van:  Aan:  Onderwerp:  Verzonden:  CC:
  English: From:  To:  Subject:   Sent:       Received:  CC:  BCC:

A new email is recognised by two complementary signals:

  1. A header field type that already appeared in the current email reappears
     (e.g. a second "Van:" means the first email has ended and a new one begins).
     Dutch and English names are normalised before comparison, so "Van:" and
     "From:" count as the same field type.

  2. A header field line appears after EXIT_BODY_THRESHOLD or more consecutive
     non-header lines — i.e. we clearly re-entered a header block from body text.

Outlook "-----Original Message-----" separators are also treated as split points.
Emails whose first visible header is only "Onderwerp:" / "Subject:" (pages that
start mid-thread, with To:/From: on the previous page) are still detected because
rule 2 fires when the next header block appears after enough body text.

Output IDs follow the pattern <doc_code>.<n>, e.g. "0003.1", "0003.2".
"""

import re

from pdf_import_reader import load_pdf

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Matches a header field line and captures the field name.
_FIELD_TYPE_RE = re.compile(
    r"^[ \t]*(?P<field>from|van|to|aan|subject|onderwerp|sent|verzonden|received|cc|bcc)\s*:",
    re.IGNORECASE,
)

# Map Dutch field names → English so "Van:" and "From:" are treated as the same.
_FIELD_NORMALIZE: dict[str, str] = {
    "van": "from", "aan": "to", "onderwerp": "subject", "verzonden": "sent",
}

# Matches Outlook "---- Original / Forwarded Message ----" separators.
_SEPARATOR_RE = re.compile(
    r"[-_=]{4,}.*?(?:original message|forwarded message|doorgestuurd bericht).*",
    re.IGNORECASE,
)

# Field extractors for metadata display.
_SUBJECT_RE = re.compile(r"^[ \t]*(?:subject|onderwerp)\s*:\s*(.+)", re.IGNORECASE)
_FROM_RE    = re.compile(r"^[ \t]*(?:from|van)\s*:\s*(.+)",           re.IGNORECASE)
_DATE_RE    = re.compile(r"^[ \t]*(?:sent|verzonden)\s*:\s*(.+)",     re.IGNORECASE)


def _normalize_field(name: str) -> str:
    return _FIELD_NORMALIZE.get(name.lower(), name.lower())


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

# After this many consecutive non-header lines we consider the header block over.
_EXIT_BODY_THRESHOLD = 6
# A header preceded by a gap of > _GAP lines AND > _BODY_LINES body lines also
# starts a new email (catches cases where the body is long but has no repeated field).
_GAP_LINES  = 4
_BODY_LINES = 2


def _find_split_points(lines: list[str]) -> list[int]:
    """
    Return line indices where new emails begin.

    For each line in document order:
    - Outlook separator  → always start a new email.
    - Header field line  → start a new email if any of:
        a) we are not currently in a header block (came from body text), OR
        b) the same field type was already seen in this email's headers, OR
        c) large line gap from last header AND several body lines seen.
      Otherwise add the field to the current email's header set.
    - Any other line     → increment body-line counter; exit header-block mode
                           once _EXIT_BODY_THRESHOLD body lines accumulate.
    """
    split_points: list[int] = []
    in_header_block = False
    seen_fields: set[str] = set()
    last_header_idx = -999
    body_lines = 0

    for i, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            split_points.append(i)
            in_header_block = True
            seen_fields = set()
            last_header_idx = i
            body_lines = 0
            continue

        m = _FIELD_TYPE_RE.match(line)
        if not m:
            if in_header_block:
                body_lines += 1
                if body_lines >= _EXIT_BODY_THRESHOLD:
                    in_header_block = False
            continue

        field = _normalize_field(m.group("field"))
        gap   = i - last_header_idx

        new_email = (
            not in_header_block                              # re-entered from body text
            or field in seen_fields                          # repeated field → new email
            or (gap > _GAP_LINES and body_lines > _BODY_LINES)  # gap + body evidence
        )

        if new_email:
            split_points.append(i)
            seen_fields = {field}
        else:
            seen_fields.add(field)

        in_header_block = True
        last_header_idx = i
        body_lines = 0

    return split_points


def _extract_field(lines: list[str], pattern: re.Pattern, limit: int = 15) -> str | None:
    """Return the first match of *pattern* in the first *limit* lines."""
    for line in lines[:limit]:
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_emails(text: str, doc_code: str) -> list[dict]:
    """
    Split the concatenated text of an e-mail document into individual emails.

    Returns a list of dicts, one per email:
        {
            "id":      str,         # e.g. "0003.2"
            "subject": str | None,
            "sender":  str | None,
            "date":    str | None,
            "text":    str,
            "warning": str | None,  # non-None when splitting was ambiguous
        }
    """
    lines = text.splitlines()
    split_points = _find_split_points(lines)

    if not split_points:
        return [{
            "id":      f"{doc_code}.1",
            "subject": None,
            "sender":  None,
            "date":    None,
            "text":    text.strip(),
            "warning": "No header-field cluster found — kept as single block",
        }]

    emails = []
    for j, start in enumerate(split_points):
        end         = split_points[j + 1] if j + 1 < len(split_points) else len(lines)
        block_lines = lines[start:end]
        block_text  = "\n".join(block_lines).strip()

        if not block_text:
            continue

        emails.append({
            "id":      f"{doc_code}.{j + 1}",
            "subject": _extract_field(block_lines, _SUBJECT_RE),
            "sender":  _extract_field(block_lines, _FROM_RE),
            "date":    _extract_field(block_lines, _DATE_RE),
            "text":    block_text,
            "warning": None,
        })

    return emails


# ---------------------------------------------------------------------------
# Main — run report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    docs = load_pdf()

    email_docs = {
        code: doc for code, doc in docs.items()
        if doc["category"] == "E-mail"
    }

    print(f"\nE-mail documents found: {list(email_docs.keys())}  ({len(email_docs)} total)")
    print("=" * 65)

    all_emails: dict[str, list[dict]] = {}
    issues: list[tuple[str, str]] = []

    for code, doc in sorted(email_docs.items()):
        emails = split_emails(doc["text"], code)
        all_emails[code] = emails

        print(f"\nDoc {code}  ({len(doc['pages'])} page(s), pages {doc['pages']})"
              f"  →  {len(emails)} individual email(s)")

        for email in emails:
            subject = email["subject"] or "(no subject)"
            sender  = email["sender"]  or "(unknown sender)"
            date    = email["date"]    or "(no date)"
            print(f"  [{email['id']}]  {subject[:70]}")
            print(f"           From : {sender[:70]}")
            print(f"           Date : {date[:50]}")
            if email["warning"]:
                print(f"           ⚠  {email['warning']}")
                issues.append((email["id"], email["warning"]))

    total = sum(len(v) for v in all_emails.values())
    print(f"\n{'=' * 65}")
    print(f"Total individual emails : {total}  across {len(email_docs)} document(s)")

    if issues:
        print(f"\nIssues ({len(issues)}):")
        for eid, msg in issues:
            print(f"  {eid}: {msg}")
    else:
        print("\nNo issues.")
