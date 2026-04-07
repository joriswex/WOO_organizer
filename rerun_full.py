"""
rerun_full.py — Full pipeline re-run (all three passes) for one or more dossiers.

Runs pass-0 (raster stamp survey + GPT-4o pilot), pass-1 (per-page GPT-4o vision),
pass-2 (LLM boundary detection for unstamped dossiers), and pass-3 (email extraction).
Overwrites the existing groundtruth cache for each dossier.

Usage:
    python rerun_full.py h                         # single dossier
    python rerun_full.py c f h                     # multiple dossiers
    python rerun_full.py a b c d e f g h i j       # all dossiers
    python rerun_full.py h --api-key sk-...        # explicit API key
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO   = Path(__file__).parent
_GT_DIR = _REPO / "groundtruth"


def rerun(letter: str, api_key: str) -> None:
    sys.path.insert(0, str(_REPO))
    from pipeline_gpt4o import load_pdf_vlm

    pdf        = _GT_DIR / f"dossier_{letter}.pdf"
    cache_path = _GT_DIR / f"dossier_{letter}_gpt4o_cache.json"

    if not pdf.exists():
        print(f"[{letter}] PDF not found: {pdf}")
        return


    print(f"[{letter}] Starting full pipeline run on {pdf.name}...")
    docs = load_pdf_vlm(pdf, api_key=api_key, cache_path=cache_path)

    print(f"\n[{letter}] Result — {len(docs)} documents:")
    for code in sorted(docs):
        doc      = docs[code]
        n_emails = len(doc.get("emails") or [])
        email_str = f"  {n_emails} email(s)" if n_emails else ""
        print(f"  {code}: {doc['category']:<18} {len(doc['pages'])} page(s){email_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full GPT-4o pipeline re-run for one or more dossiers.")
    parser.add_argument("dossiers", nargs="+", metavar="LETTER", help="Dossier letter(s), e.g. h or a b c")
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
