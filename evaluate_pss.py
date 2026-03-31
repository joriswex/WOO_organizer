"""
evaluate_pss.py
---------------
Evaluates a PSS pipeline's predictions against the OpenPSS gold standard.

USAGE
-----
    python evaluate_pss.py \
        --gold  gold_standard_sample.json \
        --pred  my_pipeline_predictions.json \
        --out   results.json

INPUT FORMAT (both --gold and --pred)
--------------------------------------
A JSON file mapping stream names to a list of per-page binary labels.

    {
      "stream_16":   [1, 0, 0, 0, 1, 0, ...],
      "stream_35":   [1, 1, 0, 1, 0, ...],
      ...
    }

Label convention (mirrors the OpenPSS gold standard):
  - 1  → this page STARTS a new document
  - 0  → this page is a CONTINUATION of the previous document
  - The first label of every stream is always 1 (trivially true) and is
    EXCLUDED from evaluation, following the OpenPSS benchmark protocol.

Your pipeline must output lists of the SAME LENGTH as the corresponding
gold standard lists, in the same stream order.

METRICS REPORTED
----------------
Per stream AND overall (macro-averaged over streams, matching OpenPSS paper):

  Page-level (boundary classification):
    - Precision, Recall, F1  (boundary class = 1)

  Document-level (Panoptic Quality / boundary F1):
    - Unweighted Document F1  (= Recognition Quality, RQ in the paper)
    - Segmentation Quality    (SQ — mean IoU of matched document pairs)
    - Weighted Document F1    (= PQ = RQ × SQ in the paper)

  These match the metrics used in:
    van Heusden et al. (2024) "OpenPSS: An Open Page Stream Segmentation Benchmark"
    https://doi.org/10.1007/978-3-031-72437-4_24
"""

import json
import argparse
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: convert label lists ↔ document segments
# ─────────────────────────────────────────────────────────────────────────────

def labels_to_segments(labels: List[int]) -> List[Tuple[int, int]]:
    """
    Convert a per-page label list to a list of (start, end) page index tuples
    (inclusive), representing individual documents within the stream.

    Example:
        [1, 0, 0, 1, 0, 1]  →  [(0,2), (3,4), (5,5)]
    """
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] == 1:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(labels) - 1))
    return segments


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Intersection over Union for two page-index segments."""
    inter_start = max(a[0], b[0])
    inter_end   = min(a[1], b[1])
    if inter_end < inter_start:
        return 0.0
    intersection = inter_end - inter_start + 1
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union


# ─────────────────────────────────────────────────────────────────────────────
# Page-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def page_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    """
    Compute boundary precision, recall, F1 at the page level.
    The first element of each list (always 1 in gold) is excluded.
    """
    assert len(gold) == len(pred), (
        f"Length mismatch: gold has {len(gold)} pages, pred has {len(pred)} pages."
    )
    g = gold[1:]
    p = pred[1:]

    tp = sum(1 for gi, pi in zip(g, p) if gi == 1 and pi == 1)
    fp = sum(1 for gi, pi in zip(g, p) if gi == 0 and pi == 1)
    fn = sum(1 for gi, pi in zip(g, p) if gi == 1 and pi == 0)

    # Edge case: no boundaries in gold AND none predicted → perfect agreement
    if tp == 0 and fp == 0 and fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": 0, "fp": 0, "fn": 0}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────────────────────────
# Document-level metrics (Panoptic Quality)
# ─────────────────────────────────────────────────────────────────────────────

def document_metrics(gold: List[int], pred: List[int]) -> Dict[str, float]:
    """
    Compute document-level Panoptic Quality metrics following Kirillov et al.
    (2019) as used by van Heusden et al. (2024).

    A true document t and predicted document p are a True Positive if IoU > 0.5.
    Each document can be part of at most one TP pair.

    Returns:
        unweighted_f1  (Recognition Quality / RQ)
        sq             (Segmentation Quality — mean IoU of TPs)
        weighted_f1    (Panoptic Quality / PQ = RQ × SQ)
    """
    true_segs = labels_to_segments(gold)
    pred_segs = labels_to_segments(pred)

    # Greedily match TPs (each segment used at most once)
    matched_true = set()
    matched_pred = set()
    iou_scores   = []

    for ti, t in enumerate(true_segs):
        for pi, p in enumerate(pred_segs):
            if pi in matched_pred:
                continue
            score = iou(t, p)
            if score > 0.5:
                matched_true.add(ti)
                matched_pred.add(pi)
                iou_scores.append(score)
                break  # each true segment gets at most one match

    tp = len(iou_scores)
    fp = len(pred_segs) - len(matched_pred)
    fn = len(true_segs) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rq        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    sq        = sum(iou_scores) / tp if tp > 0 else 0.0
    pq        = rq * sq

    return {
        "doc_precision": precision,
        "doc_recall":    recall,
        "unweighted_f1": rq,   # RQ in the paper
        "sq":            sq,
        "weighted_f1":   pq,   # PQ in the paper
        "tp": tp, "fp": fp, "fn": fn
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(gold_path: str, pred_path: str) -> Dict:
    with open(gold_path) as f:
        gold_all = json.load(f)
    with open(pred_path) as f:
        pred_all = json.load(f)

    # Only evaluate streams present in BOTH files
    common_streams = [s for s in gold_all if s in pred_all]
    missing_in_pred = [s for s in gold_all if s not in pred_all]
    extra_in_pred   = [s for s in pred_all if s not in gold_all]

    if missing_in_pred:
        print(f"[WARNING] {len(missing_in_pred)} streams in gold but not in pred "
              f"(will be skipped): {missing_in_pred}")
    if extra_in_pred:
        print(f"[WARNING] {len(extra_in_pred)} streams in pred but not in gold "
              f"(will be ignored): {extra_in_pred}")

    per_stream = {}
    for stream in common_streams:
        gold = gold_all[stream]
        pred = pred_all[stream]

        # Validate length
        if len(pred) != len(gold):
            print(f"[ERROR] Stream '{stream}': gold has {len(gold)} pages, "
                  f"pred has {len(pred)}. Skipping.")
            continue

        # Clamp predictions to 0/1 in case pipeline outputs floats
        pred = [1 if p >= 0.5 else 0 for p in pred]

        pm = page_metrics(gold, pred)
        dm = document_metrics(gold, pred)
        per_stream[stream] = {**pm, **dm, "n_pages": len(gold)}

    # Macro average over streams (matches OpenPSS paper methodology)
    def macro_avg(key):
        vals = [per_stream[s][key] for s in per_stream]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        # Page-level
        "page_precision":  macro_avg("precision"),
        "page_recall":     macro_avg("recall"),
        "page_f1":         macro_avg("f1"),
        # Document-level
        "doc_precision":   macro_avg("doc_precision"),
        "doc_recall":      macro_avg("doc_recall"),
        "unweighted_doc_f1": macro_avg("unweighted_f1"),  # RQ
        "sq":              macro_avg("sq"),
        "weighted_doc_f1": macro_avg("weighted_f1"),      # PQ
        # Counts
        "n_streams_evaluated": len(per_stream),
        "n_pages_evaluated":   sum(per_stream[s]["n_pages"] for s in per_stream),
    }

    return {"overall": overall, "per_stream": per_stream}


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print + CLI
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: Dict, pipeline_name: str = "Pipeline"):
    ov = results["overall"]
    ps = results["per_stream"]

    print(f"\n{'='*65}")
    print(f"  PSS EVALUATION RESULTS — {pipeline_name}")
    print(f"{'='*65}")
    print(f"  Streams evaluated : {ov['n_streams_evaluated']}")
    print(f"  Pages evaluated   : {ov['n_pages_evaluated']:,}")
    print(f"\n  PAGE-LEVEL METRICS (boundary classification)")
    print(f"  {'Precision':<14} {'Recall':<14} {'F1':<14}")
    print(f"  {ov['page_precision']:<14.3f} {ov['page_recall']:<14.3f} {ov['page_f1']:<14.3f}")
    print(f"\n  DOCUMENT-LEVEL METRICS (Panoptic Quality)")
    print(f"  {'Doc Precision':<16} {'Doc Recall':<14} {'Unwt. F1 (RQ)':<16} {'SQ':<10} {'Wt. F1 (PQ)'}")
    print(f"  {ov['doc_precision']:<16.3f} {ov['doc_recall']:<14.3f} "
          f"{ov['unweighted_doc_f1']:<16.3f} {ov['sq']:<10.3f} {ov['weighted_doc_f1']:.3f}")

    print(f"\n  REFERENCE (van Heusden et al. 2024, BERT+EfficientNet Late, LONG):")
    print(f"  Page F1=0.83  |  Doc F1 (RQ)=0.80  |  SQ=0.93  |  PQ=0.77")
    print(f"\n  REFERENCE (Ates 2025, Nanonets zero-shot, LONG split):")
    print(f"  Page F1=0.513 (no document-level metrics reported)")

    print(f"\n{'─'*65}")
    print(f"  PER-STREAM BREAKDOWN")
    print(f"{'─'*65}")
    print(f"  {'Stream':<45} {'Pages':>5}  {'P-F1':>5}  {'PQ':>5}  {'Rate':>5}")
    print(f"  {'─'*45}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for stream, m in sorted(ps.items(), key=lambda x: x[1]["n_pages"]):
        gold_rate = (m["tp"] + m["fn"]) / max(m["n_pages"] - 1, 1)
        short_name = stream if len(stream) <= 45 else stream[:42] + "..."
        print(f"  {short_name:<45} {m['n_pages']:>5}  "
              f"{m['f1']:>5.3f}  {m['weighted_f1']:>5.3f}  {gold_rate:>5.3f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSS pipeline predictions.")
    parser.add_argument("--gold",  required=True,
                        help="Path to gold standard JSON (stream_name -> [labels])")
    parser.add_argument("--pred",  required=True,
                        help="Path to predictions JSON (same format as gold)")
    parser.add_argument("--out",   default=None,
                        help="Optional path to write full results as JSON")
    parser.add_argument("--name",  default="My Pipeline",
                        help="Pipeline name for display")
    args = parser.parse_args()

    results = evaluate(args.gold, args.pred)
    print_results(results, pipeline_name=args.name)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to: {args.out}")
