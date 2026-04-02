"""
eval_dossier_a.py — Re-run GPT-4o pass 2 with updated prompts and compare against
ground truth annotation and OCR results for dossier_a.

Pass 1 (expensive GPT-4o vision) is loaded from the groundtruth cache — not re-run.
Only pass 2 (text-only boundary detection, cheap) is executed with the new prompts.

Usage:
    python eval_dossier_a.py
    python eval_dossier_a.py --api-key sk-...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO       = Path(__file__).parent
_GT_DIR     = _REPO / "groundtruth"
_CACHE      = _GT_DIR / "dossier_a_gpt4o_cache.json"
_OCR_SEGS   = _GT_DIR / "dossier_a_ocr_segs.json"
_ANNOTATION = _GT_DIR / "annotations_dossier_a.json"
_NEW_CACHE  = _REPO   / "dossier_a_eval_cache.json"


# ── Load ground truth ─────────────────────────────────────────────────────────

def load_gt(ann_path: Path) -> list[dict]:
    """Returns list of {start_page, end_page, type_gt, doc_index}."""
    with open(ann_path) as f:
        data = json.load(f)
    return sorted(data["documents"], key=lambda d: d["start_page"])


def load_ocr(ocr_path: Path) -> list[dict]:
    """Returns list of {start_page, end_page, doc_code, type}."""
    with open(ocr_path) as f:
        data = json.load(f)
    return sorted(data["segments"], key=lambda s: s["start_page"])


# ── Run pipeline stage 2 (mirrors actual _finalise_pipeline decision) ─────────

def run_stage2(api_key: str) -> list[dict]:
    """
    Load pass-1 page metadata from the groundtruth cache and apply the same
    decision logic as _finalise_pipeline:
      - Stamps on >= _STAMP_THRESHOLD pages → stamp forward-fill (no LLM call)
      - Otherwise → pass-2 LLM boundary detection

    Returns a list of {start_page (0-based), end_page, doc_code} dicts.
    """
    sys.path.insert(0, str(_REPO))
    from pipeline_gpt4o import (
        _load_cache_pages,
        _build_page_summary,
        _call_boundary_pass,
        _update_cache_boundaries,
        _normalise_doc_code,
        _STAMP_THRESHOLD,
    )
    import shutil
    from openai import OpenAI

    shutil.copy2(_CACHE, _NEW_CACHE)
    pages_meta, _dpi, _old_boundaries = _load_cache_pages(_NEW_CACHE)
    n = len(pages_meta)
    print(f"[eval] Loaded {n} pages from pass-1 cache.")

    stamped  = sum(1 for p in pages_meta if p.get("doc_code"))
    coverage = stamped / n if n else 0.0
    print(f"[eval] Stamp coverage: {stamped}/{n} pages ({coverage:.0%})  threshold={_STAMP_THRESHOLD:.0%}")

    if coverage >= _STAMP_THRESHOLD:
        print("[eval] Reliable stamps → using stamp forward-fill (no LLM pass-2 needed).")
        return _stamp_forward_fill(pages_meta, n)

    print("[eval] Sparse stamps → running LLM pass-2...")
    client    = OpenAI(api_key=api_key)
    summary   = _build_page_summary(pages_meta)
    documents = _call_boundary_pass(summary, n, client)

    if documents:
        print(f"[eval] Pass 2 returned {len(documents)} documents.")
        _update_cache_boundaries(documents, _NEW_CACHE)
        # Convert 1-based pass-2 start_page to 0-based segments
        docs_sorted = sorted(documents, key=lambda d: d["start_page"])
        result = []
        for i, doc in enumerate(docs_sorted):
            start = int(doc["start_page"]) - 1
            end   = (int(docs_sorted[i + 1]["start_page"]) - 2
                     if i + 1 < len(docs_sorted) else n - 1)
            result.append({"start_page": start, "end_page": end, "doc_code": doc.get("doc_code")})
        return result

    print("[eval] WARNING: pass 2 returned no documents.")
    return []


def _stamp_forward_fill(pages_meta: list[dict], total: int) -> list[dict]:
    """Replicate stamp forward-fill without needing page images (eval only)."""
    sys.path.insert(0, str(_REPO))
    from pipeline_gpt4o import _normalise_doc_code

    last_code     = None
    unknown_count = 0
    segs: dict    = {}

    for p in pages_meta:
        raw  = p.get("doc_code")
        code = _normalise_doc_code(raw) if raw else None
        wpn  = p.get("within_doc_page")
        is_new = (p.get("is_new_document", False) or wpn == 1) and (wpn is None or wpn == 1)
        pg   = p["page_num"] - 1  # 0-indexed

        if code:
            current = code; last_code = code
        elif is_new and last_code is not None:
            unknown_count += 1; current = f"unknown_{unknown_count}"; last_code = current
        else:
            if last_code is None:
                unknown_count += 1; last_code = f"unknown_{unknown_count}"
            current = last_code

        if current not in segs:
            segs[current] = {"doc_code": current, "pages": []}
        segs[current]["pages"].append(pg)

    return sorted(
        [{"start_page": min(v["pages"]), "end_page": max(v["pages"]), "doc_code": v["doc_code"]}
         for v in segs.values()],
        key=lambda s: s["start_page"],
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _boundary_set(segs: list[dict]) -> set[int]:
    """Set of start_page values — used for boundary overlap scoring."""
    return {s["start_page"] for s in segs}


def score_boundaries(pred: list[dict], gt: list[dict]) -> dict:
    """
    WindowDiff-style boundary scoring.
    Returns precision, recall, F1, and a count of exact boundary matches.
    A boundary is a hit if it falls within ±1 page of any GT boundary.
    """
    gt_starts  = sorted(_boundary_set(gt))
    pred_starts = sorted(_boundary_set(pred))

    # Count hits (within ±1 page tolerance)
    hit_gt   = set()
    hit_pred = set()
    for ps in pred_starts:
        for gs in gt_starts:
            if abs(ps - gs) <= 1:
                hit_gt.add(gs)
                hit_pred.add(ps)
                break

    tp = len(hit_pred)
    fp = len(pred_starts) - tp
    fn = len(gt_starts)  - len(hit_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "pred_docs":  len(pred_starts),
        "gt_docs":    len(gt_starts),
        "tp":         tp,
        "fp":         fp,
        "fn":         fn,
        "precision":  round(precision, 3),
        "recall":     round(recall, 3),
        "f1":         round(f1, 3),
    }


# ── Side-by-side table ────────────────────────────────────────────────────────

def _find_doc_for_page(segs: list[dict], page: int) -> dict | None:
    for s in segs:
        if s["start_page"] <= page <= s["end_page"]:
            return s
    return None


def print_comparison_table(gt_docs: list[dict],
                            gpt4o_new: list[dict],
                            ocr_segs: list[dict]) -> None:
    HDR = f"{'#':>3}  {'GT start':>8}  {'GT end':>6}  {'GT type':>16}  " \
          f"{'GPT4o start':>11}  {'GPT4o end':>9}  {'GPT4o code':>10}  " \
          f"{'OCR start':>9}  {'OCR end':>7}  {'OCR code':>8}  {'OCR type':>16}"
    print(HDR)
    print("-" * len(HDR))

    for d in gt_docs:
        sp    = d["start_page"]
        ep    = d["end_page"]
        tgt   = d["type_gt"]

        # Find the GPT-4o new doc that covers this GT start page
        g = _find_doc_for_page(gpt4o_new, sp)
        g_start = g["start_page"] if g else "?"
        g_end   = g["end_page"]   if g else "?"
        g_code  = g["doc_code"]   if g else "?"
        g_match = "✓" if g and g["start_page"] == sp else "~" if g else "✗"

        # Find the OCR doc
        o = _find_doc_for_page(ocr_segs, sp)
        o_start = o["start_page"] if o else "?"
        o_end   = o["end_page"]   if o else "?"
        o_code  = o["doc_code"]   if o else "?"
        o_type  = o["type"]       if o else "?"
        o_match = "✓" if o and o["start_page"] == sp else "~" if o else "✗"

        print(f"{d['doc_index']:>3}  {sp:>8}  {ep:>6}  {tgt:>16}  "
              f"{g_match}{str(g_start):>10}  {str(g_end):>9}  {str(g_code):>10}  "
              f"{o_match}{str(o_start):>8}  {str(o_end):>7}  {str(o_code):>8}  {o_type:>16}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o pass 2 vs GT and OCR")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: No OpenAI API key. Set OPENAI_API_KEY or pass --api-key.")
        sys.exit(1)

    gt_docs  = load_gt(_ANNOTATION)
    ocr_segs = load_ocr(_OCR_SEGS)
    total_pages = max(d["end_page"] for d in gt_docs) + 1

    # Run stage 2 (stamp forward-fill or LLM pass-2 depending on stamp coverage)
    gpt4o_new = run_stage2(api_key)

    # Scores
    print("\n" + "=" * 70)
    print("BOUNDARY DETECTION SCORES  (±1 page tolerance)")
    print("=" * 70)
    s_gpt = score_boundaries(gpt4o_new, gt_docs)
    s_ocr = score_boundaries(ocr_segs,  gt_docs)

    print(f"\n{'Pipeline':>20}  {'Pred':>4}  {'GT':>4}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
          f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("-" * 70)
    print(f"{'GPT-4o (new prompts)':>20}  {s_gpt['pred_docs']:>4}  {s_gpt['gt_docs']:>4}  "
          f"{s_gpt['tp']:>4}  {s_gpt['fp']:>4}  {s_gpt['fn']:>4}  "
          f"{s_gpt['precision']:>6.3f}  {s_gpt['recall']:>6.3f}  {s_gpt['f1']:>6.3f}")
    print(f"{'OCR':>20}  {s_ocr['pred_docs']:>4}  {s_ocr['gt_docs']:>4}  "
          f"{s_ocr['tp']:>4}  {s_ocr['fp']:>4}  {s_ocr['fn']:>4}  "
          f"{s_ocr['precision']:>6.3f}  {s_ocr['recall']:>6.3f}  {s_ocr['f1']:>6.3f}")

    print("\n" + "=" * 70)
    print("DOCUMENT-BY-DOCUMENT COMPARISON  (✓=exact match  ~=off by ≤1 page  ✗=missed)")
    print("=" * 70 + "\n")
    print_comparison_table(gt_docs, gpt4o_new, ocr_segs)

    # Save boundary result for inspection
    out = _REPO / "dossier_a_gpt4o_new_boundaries.json"
    with open(out, "w") as f:
        json.dump({"gpt4o_new": gpt4o_new}, f, indent=2)
    print(f"\n[eval] Boundary result saved → {out}")


if __name__ == "__main__":
    main()
