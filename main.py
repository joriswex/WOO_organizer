"""
main.py — full WOO document pipeline.

Steps:
  1. pdf_import_reader  — extract text and metadata from the PDF
  2. text_sorting       — detect each document's date and sort chronologically
                          (email_splitter is used internally for e-mail documents)
  3. visualisation      — render an interactive HTML timeline
"""
from pathlib import Path

from pdf_import_reader import load_pdf
from text_sorting import sort_documents
from visualisation import build_html

PDF_PATH = Path("test.pdf")
OUT_PATH = Path("woo_timeline.html")


def main() -> None:
    print("Step 1/3 — Loading and parsing PDF (with 300 DPI OCR supplement)...")
    docs = load_pdf(PDF_PATH, ocr_supplement=True)

    print("Step 2/3 — Extracting dates and sorting chronologically...")
    docs = sort_documents(docs)

    print("Step 3/3 — Building interactive HTML timeline...")
    build_html(docs, OUT_PATH, pdf_path=PDF_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
