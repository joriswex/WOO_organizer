"""
main_gpt4o.py — WOO pipeline using GPT-4o full-page VLM text extraction.

Steps:
  1. pipeline_gpt4o  — extract text AND classify each page via GPT-4o vision
  2. text_sorting    — detect each document's date and sort chronologically
  3. visualisation   — render an interactive HTML timeline

Usage:
    python main_gpt4o.py --pdf palestina_combined.pdf       # full run, auto-saves cache
    python main_gpt4o.py --pdf palestina_combined.pdf --vlm-pages 15   # test: first 15 pages
    python main_gpt4o.py --from-cache palestina_combined_cache.json    # regenerate HTML only
    python main_gpt4o.py --api-key sk-...                   # explicit OpenAI key
                                                            # (or set OPENAI_API_KEY env var)
"""

import argparse
import os
from pathlib import Path

from pipeline_gpt4o import load_pdf_vlm, docs_from_cache
from text_sorting import sort_documents
from visualisation import build_html

_DEFAULT_PDF = Path("test.pdf")
_DEFAULT_OUT = Path("woo_timeline_vlm.html")


def _default_cache_path(pdf_path: Path) -> Path:
    return pdf_path.with_name(pdf_path.stem + "_gpt4o_cache.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WOO pipeline — GPT-4o full VLM text extraction + HTML timeline"
    )
    parser.add_argument(
        "--pdf", default=str(_DEFAULT_PDF), metavar="FILE",
        help="PDF file to process (default: test.pdf)",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT), metavar="FILE",
        help="Output HTML file (default: woo_timeline_vlm.html)",
    )
    parser.add_argument(
        "--vlm-pages", type=int, default=None, metavar="N",
        help="Only process the first N pages — useful for testing quality before a full run",
    )
    parser.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="OpenAI API key (falls back to OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--from-cache", default=None, metavar="FILE",
        help="Skip API calls — rebuild HTML from a previously saved JSON cache file",
    )
    args = parser.parse_args()

    pdf_path   = Path(args.pdf)
    out_path   = Path(args.out)
    from_cache = Path(args.from_cache) if args.from_cache else None

    # ── Rebuild from cache (no API needed) ────────────────────────────────────
    if from_cache:
        if not from_cache.exists():
            print(f"ERROR: Cache file not found: {from_cache}")
            return
        if not pdf_path.exists():
            print(f"ERROR: PDF not found (needed for page images): {pdf_path}")
            return

        print(f"Rebuilding from cache: {from_cache}")
        docs = docs_from_cache(from_cache, pdf_path)

        print("\nStep 2/3 — Extracting dates and sorting chronologically...")
        docs = sort_documents(docs)

        print("\nStep 3/3 — Building interactive HTML timeline...")
        build_html(docs, out_path, pdf_path=pdf_path)

        print(f"\nDone. Output: {out_path}")
        return

    # ── Full API run ───────────────────────────────────────────────────────────
    api_key   = args.api_key or os.environ.get("OPENAI_API_KEY")
    max_pages = args.vlm_pages

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return

    if not api_key:
        print(
            "ERROR: No OpenAI API key.\n"
            "  Set it with:  export OPENAI_API_KEY=sk-...\n"
            "  Or pass it:   python main_gpt4o.py --api-key sk-..."
        )
        return

    cache_path = _default_cache_path(pdf_path)
    if max_pages:
        # Test runs get a separate cache so they don't overwrite the full-run cache
        cache_path = pdf_path.with_name(f"{pdf_path.stem}_gpt4o_cache_p{max_pages}.json")
        print(f"Test mode — analysing first {max_pages} pages only.")

    print(f"Cache will be saved to: {cache_path}")
    print(f"Output will go to:      {out_path}\n")

    print("Step 1/3 — GPT-4o full-page VLM analysis (text extraction + classification)...")
    docs = load_pdf_vlm(pdf_path, api_key=api_key, max_pages=max_pages, cache_path=cache_path)

    print("\nStep 2/3 — Extracting dates and sorting chronologically...")
    docs = sort_documents(docs)

    print("\nStep 3/3 — Building interactive HTML timeline...")
    build_html(docs, out_path, pdf_path=pdf_path)

    print(f"\nDone. Output: {out_path}")
    print(f"To regenerate HTML without re-running the API:")
    print(f"  python main_gpt4o.py --pdf {pdf_path} --from-cache {cache_path}")


if __name__ == "__main__":
    main()
