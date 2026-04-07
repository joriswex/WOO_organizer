"""
rerun_pass2.py — Re-run pass-2 (boundary detection) and pass-3 (email extraction)
for a dossier from its existing pass-1 cache. Pass-1 (expensive GPT-4o vision) is
NOT re-run; only the cheap text-only passes are executed with updated prompts.

Usage:
    python rerun_pass2.py h                      # uses OPENAI_API_KEY env var
    python rerun_pass2.py h --api-key sk-...
    python rerun_pass2.py a b g h --api-key sk-...  # multiple dossiers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO   = Path(__file__).parent
_GT_DIR = _REPO / "groundtruth"


def rerun(letter: str, api_key: str) -> None:
    sys.path.insert(0, str(_REPO))
    from pdf2image import convert_from_path
    from openai import OpenAI
    from pipeline_gpt4o import _load_cache_pages, _finalise_pipeline

    cache = _GT_DIR / f"dossier_{letter}_gpt4o_cache.json"
    pdf   = _GT_DIR / f"dossier_{letter}.pdf"

    if not cache.exists():
        print(f"[{letter}] Cache not found: {cache}")
        return
    if not pdf.exists():
        print(f"[{letter}] PDF not found: {pdf}")
        return

    # Strip stale boundary decisions so _finalise_pipeline re-runs pass-2 fresh
    with open(cache) as f:
        data = json.load(f)
    had_boundaries = "boundary_documents" in data
    had_emails     = "emails_by_doc" in data
    data.pop("boundary_documents", None)
    data.pop("emails_by_doc", None)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[{letter}] Cache cleared (boundary_documents={had_boundaries}, emails_by_doc={had_emails}).")

    # Load pass-1 metadata + render page images from PDF
    pages_meta, dpi, _ = _load_cache_pages(cache)
    n = len(pages_meta)
    print(f"[{letter}] Rendering {n} pages at {dpi} DPI from {pdf.name}...")
    images = convert_from_path(str(pdf), dpi=dpi)
    page_data = [
        {**p, "image": images[p["page_num"] - 1]}
        for p in pages_meta
        if p["page_num"] - 1 < len(images)
    ]

    # Re-run pass-2 (with inventarislijst hint) and pass-3 (email extraction)
    client = OpenAI(api_key=api_key)
    docs   = _finalise_pipeline(page_data, client=client, cache_path=cache)

    print(f"\n[{letter}] Result — {len(docs)} documents:")
    for code in sorted(docs):
        doc      = docs[code]
        n_emails = len(doc.get("emails") or [])
        email_str = f"  {n_emails} email(s)" if n_emails else ""
        print(f"  {code}: {doc['category']:<18} {len(doc['pages'])} page(s){email_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run GPT-4o pass-2 and pass-3 from cached pass-1 data.")
    parser.add_argument("dossiers", nargs="+", metavar="LETTER", help="Dossier letter(s), e.g. h or a b g h")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set OPENAI_API_KEY.")
        sys.exit(1)

    for letter in args.dossiers:
        print(f"\n{'=' * 60}")
        print(f"DOSSIER {letter.upper()}")
        print(f"{'=' * 60}")
        rerun(letter.lower(), args.api_key)


if __name__ == "__main__":
    main()
