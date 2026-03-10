"""
main.py — full WOO document pipeline.

Steps:
  1. pdf_import_reader  — extract text and metadata from the PDF
  2. text_sorting       — detect each document's date and sort chronologically
                          (email_splitter is used internally for e-mail documents)
  3. visualisation      — render an interactive HTML timeline

Usage:
    python main.py                        # process test.pdf → woo_timeline.html
    python main.py --vlm                  # with VLM assist (requires Ollama)
    python main.py --pdf other.pdf --vlm  # custom PDF with VLM
    python main.py --pdf other.pdf --out out.html
"""
import argparse
from pathlib import Path

from pdf_import_reader import load_pdf
from text_sorting import sort_documents
from visualisation import build_html

_DEFAULT_PDF = Path("test.pdf")
_DEFAULT_OUT = Path("woo_timeline.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WOO document pipeline — PDF → interactive HTML timeline"
    )
    parser.add_argument(
        "--pdf", default=str(_DEFAULT_PDF), metavar="FILE",
        help="PDF file to process (default: test.pdf)",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT), metavar="FILE",
        help="Output HTML file (default: woo_timeline.html)",
    )
    parser.add_argument(
        "--vlm", action="store_true",
        help="Enable VLM assist via Ollama/Qwen2.5-VL for better boundary detection",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    vlm_on   = args.vlm

    # ── VLM availability check ────────────────────────────────────────────────
    if vlm_on:
        from vlm_classifier import is_ollama_available, DEFAULT_MODEL
        if not is_ollama_available(DEFAULT_MODEL):
            print(
                "[vlm] WARNING: Ollama is not running or qwen2.5vl:7b is not pulled.\n"
                "[vlm]   Start Ollama: ollama serve\n"
                "[vlm]   Pull model:   ollama pull qwen2.5vl:7b\n"
                "[vlm] Falling back to non-VLM mode."
            )
            vlm_on = False
        else:
            print(f"[vlm] Ollama available — VLM assist enabled ({DEFAULT_MODEL})")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print("Step 1/3 — Loading and parsing PDF (with 300 DPI OCR supplement)...")
    docs = load_pdf(pdf_path, ocr_supplement=True, vlm_assist=vlm_on)

    print("Step 2/3 — Extracting dates and sorting chronologically...")
    docs = sort_documents(docs)

    print("Step 3/3 — Building interactive HTML timeline...")
    build_html(docs, out_path, pdf_path=pdf_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
