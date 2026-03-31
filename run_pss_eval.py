"""
run_pss_eval.py — PSS (Page Stream Segmentation) evaluation harness.

Runs either the VLM (GPT-4o) or OCR (Tesseract) pipeline over the streams
in gold_standard_sample.json and evaluates boundary-detection accuracy against
the OpenPSS benchmark.

Usage:
    python run_pss_eval.py --pipeline vlm --out predictions_vlm.json
    python run_pss_eval.py --pipeline ocr --out predictions_ocr.json
    python run_pss_eval.py --pipeline vlm --dry-run
    python run_pss_eval.py --pipeline ocr --dry-run

Stream–file mapping:
    Each stream name in gold_standard_sample.json maps directly to PNG files
    in the 'OpenPSS Evaluation/png/' directory:
        stream_16  →  stream_16-1.png, stream_16-2.png, …, stream_16-50.png
    Page numbers are 1-based.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent
_GOLD_PATH = _REPO_ROOT / "gold_standard_sample.json"
_PNG_DIR   = _REPO_ROOT / "OpenPSS Evaluation" / "png"

_DRY_RUN_STREAMS = 2
_DRY_RUN_PAGES   = 20


# ── OCR pipeline ───────────────────────────────────────────────────────────────

def _run_ocr_stream(stream_name: str, n_pages: int) -> list[int]:
    """
    Run the OCR (Tesseract) pipeline on one stream.

    Loads each page image from the PNG directory, runs full-page OCR and
    raster-based stamp detection, then applies the same boundary heuristics
    used internally by pipeline_ocr.py.

    When doc-code stamps are detected anywhere in the stream, code changes
    drive the boundary decisions (matching the pipeline's forward-fill mode).
    When no stamps are found, _auto_split_boundaries() handles the prediction
    via text heuristics (page counters, email headers, nota headers).
    """
    from PIL import Image
    from pipeline_ocr import (
        _auto_split_boundaries,
        _find_doc_code_raster,
        _find_within_doc_page_raster,
        _is_new_doc_boundary,
        _ocr_image,
    )

    page_data: list[dict] = []

    for i in range(1, n_pages + 1):
        png_path = _PNG_DIR / f"{stream_name}-{i}.png"
        if not png_path.exists():
            print(f"  [ocr] Warning: {png_path.name} not found — treating as continuation")
            page_data.append({"text": "", "within_doc_page": None, "detected_code": None})
            continue

        img             = Image.open(png_path)
        text            = _ocr_image(img)
        within_doc_page = _find_within_doc_page_raster(img)
        doc_code        = _find_doc_code_raster(img)
        page_data.append({
            "text":            text,
            "within_doc_page": within_doc_page,
            "detected_code":   doc_code,
        })

    n = len(page_data)
    predictions = [0] * n
    predictions[0] = 1  # first page always starts a document

    has_stamps = any(p["detected_code"] for p in page_data)

    if has_stamps:
        # Primary signal: doc code change from the last known code.
        # Secondary: within_doc_page == 1 or "Pagina 1 van" in text.
        last_code: str | None = None
        for i, p in enumerate(page_data):
            code = p["detected_code"]
            if i == 0:
                if code:
                    last_code = code
                continue
            if code and code != last_code:
                predictions[i] = 1
                last_code = code
            elif _is_new_doc_boundary(p):
                predictions[i] = 1
    else:
        # No stamps found — use text-based auto-split heuristics.
        boundaries = _auto_split_boundaries(page_data)
        predictions = [1 if b else 0 for b in boundaries]

    return predictions


# ── VLM pipeline ───────────────────────────────────────────────────────────────

def _run_vlm_stream(stream_name: str, n_pages: int, client) -> list[int]:
    """
    Run the VLM (GPT-4o) pipeline on one stream.

    Loads each page image from the PNG directory, encodes it, and calls
    GPT-4o vision using the same prompt and encoding logic as pipeline_gpt4o.py.
    API calls for pages 2..N are issued in parallel (up to _MAX_WORKERS threads).
    Page 1 is always set to 1 without an API call.
    """
    from PIL import Image
    from pipeline_gpt4o import _MAX_WORKERS, _call_gpt4o, _encode_image

    predictions = [0] * n_pages
    predictions[0] = 1  # first page always starts a document

    def _process_page(args: tuple[int, Path]) -> tuple[int, int]:
        i, png_path = args
        if not png_path.exists():
            print(f"  [vlm] Warning: {png_path.name} not found — treating as continuation")
            return i, 0
        img       = Image.open(png_path)
        image_b64 = _encode_image(img)
        result    = _call_gpt4o(image_b64, client, page_index=i, pdf_page_num=i + 1)
        is_new    = result.get("is_new_document", False)
        return i, (1 if is_new else 0)

    # Pages 1..n_pages-1 (0-based), skipping page 0 (always 1, no API call)
    tasks = [
        (i, _PNG_DIR / f"{stream_name}-{i + 1}.png")
        for i in range(1, n_pages)
    ]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_process_page, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            i, label = fut.result()
            predictions[i] = label

    return predictions


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PSS pipeline evaluation against gold_standard_sample.json"
    )
    parser.add_argument(
        "--pipeline", choices=["vlm", "ocr"], required=True,
        help="Pipeline to evaluate: 'vlm' (GPT-4o) or 'ocr' (Tesseract)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output predictions JSON path (default: predictions_{pipeline}[_dryrun].json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            f"Process only the first {_DRY_RUN_STREAMS} streams × "
            f"{_DRY_RUN_PAGES} pages each (end-to-end sanity check)"
        ),
    )
    parser.add_argument(
        "--streams", type=int, default=None, metavar="N",
        help="Process only the first N streams in full and evaluate (e.g. --streams 3 ≈ 93 pages)",
    )
    parser.add_argument(
        "--gold", default=str(_GOLD_PATH),
        help=f"Gold standard JSON path (default: {_GOLD_PATH.name})",
    )
    args = parser.parse_args()

    # ── Resolve output paths ──────────────────────────────────────────────────
    gold_path = Path(args.gold)
    dry_suffix = "_dryrun" if args.dry_run else ""
    streams_suffix = f"_s{args.streams}" if args.streams is not None else ""

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _REPO_ROOT / f"predictions_{args.pipeline}{dry_suffix}{streams_suffix}.json"

    stem = out_path.stem
    results_stem = (
        "results_" + stem[len("predictions_"):]
        if stem.startswith("predictions_")
        else stem + "_results"
    )
    results_path = out_path.parent / (results_stem + ".json")

    pipeline_label = "GPT-4o VLM" if args.pipeline == "vlm" else "Tesseract OCR"

    # ── Load gold standard ────────────────────────────────────────────────────
    with open(gold_path) as f:
        gold: dict[str, list[int]] = json.load(f)

    if args.dry_run:
        print(
            "\n*** DRY-RUN MODE ***\n"
            f"Processing first {_DRY_RUN_STREAMS} streams × "
            f"{_DRY_RUN_PAGES} pages each.\n"
            "Results are NOT comparable to the benchmark.\n"
        )
        gold = dict(list(gold.items())[:_DRY_RUN_STREAMS])
    elif args.streams is not None:
        gold = dict(list(gold.items())[:args.streams])
        n_pages_total = sum(len(v) for v in gold.values())
        print(
            f"\n*** PARTIAL RUN: first {args.streams} stream(s), "
            f"{n_pages_total} pages total ***\n"
            "Scores are real (no padding) but cover a subset of the benchmark.\n"
        )

    # ── Set up VLM client ─────────────────────────────────────────────────────
    client = None
    if args.pipeline == "vlm":
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package not installed.  Run: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            sys.exit("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)

    # ── Run pipeline stream by stream ─────────────────────────────────────────
    predictions: dict[str, list[int]] = {}
    start_total = time.time()

    for idx, (stream_name, gold_labels) in enumerate(gold.items()):
        full_n = len(gold_labels)
        n_pages = min(full_n, _DRY_RUN_PAGES) if args.dry_run else full_n

        print(f"\n[{idx + 1}/{len(gold)}] Stream '{stream_name}' ({n_pages} pages)")
        stream_start = time.time()

        try:
            if args.pipeline == "vlm":
                page_preds = _run_vlm_stream(stream_name, n_pages, client)
            else:
                page_preds = _run_ocr_stream(stream_name, n_pages)
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping stream '{stream_name}'")
            continue

        # In dry-run the predictions are shorter than the full gold list.
        # Pad with 0s so format validation can still check lengths.
        if len(page_preds) < full_n:
            page_preds = page_preds + [0] * (full_n - len(page_preds))

        assert page_preds[0] == 1, (
            f"Stream '{stream_name}': first prediction must be 1 (got {page_preds[0]})"
        )
        assert len(page_preds) == full_n, (
            f"Stream '{stream_name}': expected {full_n} predictions, got {len(page_preds)}"
        )

        elapsed = time.time() - stream_start
        print(f"  Done in {elapsed:.1f}s  ({elapsed / n_pages:.2f}s/page)")

        predictions[stream_name] = page_preds

        # Incremental save so a crash does not lose completed streams
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)

    total = time.time() - start_total
    print(f"\nTotal time: {total / 60:.1f} minutes")
    print(f"Predictions written to: {out_path}")

    if not predictions:
        print("No streams were successfully processed — nothing to evaluate.")
        return

    # ── Format validation ─────────────────────────────────────────────────────
    print("\n--- Format validation ---")
    from pipeline_output_template import validate

    if args.dry_run or args.streams is not None:
        # Validate only against the streams we actually predicted so that
        # missing-stream errors do not mask real format problems.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=_REPO_ROOT
        ) as tmp:
            json.dump(gold, tmp, indent=2)
            tmp_gold_path = tmp.name
        try:
            validate(tmp_gold_path, str(out_path))
        finally:
            os.unlink(tmp_gold_path)
    else:
        validate(str(gold_path), str(out_path))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(
            "\n[dry-run] Skipping full evaluation — predictions for pages "
            f"{_DRY_RUN_PAGES + 1}+ are padded zeros, so scores would be meaningless."
        )
        return

    print("\n--- Evaluation ---")
    from evaluate_pss import evaluate, print_results

    # evaluate() only scores streams present in both files; extra gold streams
    # trigger a warning but do not affect scores — fine for partial runs.
    results = evaluate(str(gold_path), str(out_path))
    print_results(results, pipeline_name=pipeline_label)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to: {results_path}")


if __name__ == "__main__":
    main()
