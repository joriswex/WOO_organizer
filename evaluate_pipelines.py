"""
Pipeline evaluation: OCR vs GPT-4o across annotated dossiers.
Compares boundary detection and document type classification.

Auto-detects dossiers by scanning for annotations_dossier_X.json files.
Skips any dossier that is missing either the OCR segs or GPT-4o cache in groundtruth/.
"""
import json
from collections import Counter
from pathlib import Path

GT_DIR = Path("groundtruth")
ROOT = Path(".")


def _discover_dossiers() -> list[str]:
    """Return sorted list of dossier letters that have all three required files."""
    letters = []
    for ann in sorted(ROOT.glob("annotations_dossier_?.json")):
        letter = ann.stem.split("_")[-1]        # "annotations_dossier_a" → "a"
        ocr   = GT_DIR / f"dossier_{letter}_ocr_segs.json"
        gpt4o = GT_DIR / f"dossier_{letter}_gpt4o_cache.json"
        if ocr.exists() and gpt4o.exists():
            letters.append(letter)
        else:
            missing = [f for f, p in [("ocr_segs", ocr), ("gpt4o_cache", gpt4o)] if not p.exists()]
            print(f"[eval] skipping dossier {letter}: missing {', '.join(missing)}")
    return letters


DOSSIERS = _discover_dossiers()

# Map GPT-4o / OCR types to a canonical set for fair comparison
TYPE_MAP = {
    # OCR variants
    "Other": "Other",
    "Nota": "Nota",
    "E-mail": "E-mail",
    "Brief": "Brief",
    "Report": "Report",
    "Timeline": "Timeline",
    "Vergadernotulen": "Vergadernotulen",
    # GPT-4o-only
    "Chat": "Chat",
    # GT can be empty string
    "": "Other",
}


def load_gt(dossier: str):
    path = ROOT / f"annotations_dossier_{dossier}.json"
    with open(path) as f:
        data = json.load(f)
    return data["boundaries"], data["annotations"]


def load_ocr(dossier: str):
    path = GT_DIR / f"dossier_{dossier}_ocr_segs.json"
    with open(path) as f:
        return json.load(f)["segments"]


def load_gpt4o(dossier: str):
    path = GT_DIR / f"dossier_{dossier}_gpt4o_cache.json"
    with open(path) as f:
        return json.load(f)


def ocr_boundaries(segments):
    """Return sorted unique set of 0-indexed start pages."""
    return sorted({s["start_page"] for s in segments})


def gpt4o_boundaries(pages):
    """
    Return sorted unique set of 0-indexed start pages.
    page_num is 1-indexed; is_new_document=True marks a segment start.
    Page 0 is always implicitly a boundary.
    """
    boundaries = set()
    for p in pages:
        if p.get("is_new_document"):
            boundaries.add(p["page_num"] - 1)  # convert to 0-indexed
    # First page is always a boundary
    if pages:
        boundaries.add(0)
    return sorted(boundaries)


def boundary_metrics(predicted: list, ground_truth: list, tolerance: int = 0):
    """Precision / Recall / F1 for boundary detection."""
    pred_set = set(predicted)
    gt_set = set(ground_truth)

    if tolerance == 0:
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
    else:
        matched_gt = set()
        matched_pred = set()
        for p in pred_set:
            for g in gt_set:
                if abs(p - g) <= tolerance and g not in matched_gt:
                    matched_gt.add(g)
                    matched_pred.add(p)
                    break
        tp = len(matched_gt)
        fp = len(pred_set) - len(matched_pred)
        fn = len(gt_set) - len(matched_gt)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def gpt4o_type_for_segment(pages, start_page_0indexed, end_page_0indexed):
    """
    Majority category across pages in [start, end] (0-indexed, inclusive).
    page_num in pages is 1-indexed.
    """
    cats = []
    for p in pages:
        pg = p["page_num"] - 1  # 0-indexed
        if start_page_0indexed <= pg <= end_page_0indexed:
            cat = p.get("category", "Other") or "Other"
            cats.append(TYPE_MAP.get(cat, cat))
    if not cats:
        return "Other"
    return Counter(cats).most_common(1)[0][0]


def ocr_type_for_segment(ocr_segments, start_page_0indexed):
    """
    Find the OCR segment whose range contains start_page, return its type.
    Falls back to the segment with start_page nearest and <= query.
    """
    best = None
    for seg in ocr_segments:
        if seg["start_page"] <= start_page_0indexed <= seg["end_page"]:
            return TYPE_MAP.get(seg["type"], seg["type"])
        if seg["start_page"] <= start_page_0indexed:
            best = seg
    if best:
        return TYPE_MAP.get(best["type"], best["type"])
    return "Other"


def type_accuracy(gt_boundaries, gt_annotations, ocr_segments, gpt4o_pages, total_pages):
    """
    For each GT segment, compare OCR and GPT-4o predicted types.
    Returns per-segment results and aggregate accuracy.
    """
    results = []
    for i, start in enumerate(gt_boundaries):
        end = gt_boundaries[i + 1] - 1 if i + 1 < len(gt_boundaries) else total_pages - 1
        ann = gt_annotations.get(str(start), {})
        gt_type = TYPE_MAP.get(ann.get("type_gt", ""), "Other")

        ocr_type = ocr_type_for_segment(ocr_segments, start)
        gpt_type = gpt4o_type_for_segment(gpt4o_pages, start, end)

        results.append({
            "start": start,
            "end": end,
            "gt": gt_type,
            "ocr": ocr_type,
            "gpt4o": gpt_type,
            "ocr_ok": ocr_type == gt_type,
            "gpt4o_ok": gpt_type == gt_type,
        })
    ocr_acc  = sum(r["ocr_ok"]   for r in results) / len(results) if results else 0
    gpt4o_acc = sum(r["gpt4o_ok"] for r in results) / len(results) if results else 0
    return results, ocr_acc, gpt4o_acc


def email_count_accuracy(gt_boundaries, gt_annotations, ocr_segments, gpt4o_data, total_pages):
    """Compare email sub-document counts for Email-typed GT segments."""
    results = []
    emails_by_doc = gpt4o_data.get("emails_by_doc", {})
    gpt4o_pages  = gpt4o_data["pages"]

    for i, start in enumerate(gt_boundaries):
        end = gt_boundaries[i + 1] - 1 if i + 1 < len(gt_boundaries) else total_pages - 1
        ann = gt_annotations.get(str(start), {})
        if ann.get("type_gt") != "E-mail":
            continue
        gt_count = ann.get("num_emails_gt")
        if gt_count is None:
            continue

        # OCR: count emails in OCR segment spanning this start page
        ocr_emails = None
        for seg in ocr_segments:
            if seg["start_page"] <= start <= seg["end_page"]:
                ocr_emails = seg.get("num_emails", None)
                break

        # GPT-4o: look up emails_by_doc using doc_code from first page of segment
        first_page = next(
            (p for p in gpt4o_pages if p["page_num"] - 1 == start), None
        )
        gpt_emails = None
        if first_page and first_page.get("doc_code"):
            code = first_page["doc_code"]
            entries = emails_by_doc.get(code, [])
            gpt_emails = len(entries) if entries else None

        results.append({
            "start": start,
            "gt_count": gt_count,
            "ocr_count": ocr_emails,
            "gpt4o_count": gpt_emails,
        })
    return results


def run():
    if not DOSSIERS:
        print("No dossiers found. Add annotations_dossier_X.json + groundtruth/dossier_X_{ocr_segs,gpt4o_cache}.json")
        return

    print("=" * 72)
    print(f"WOO PIPELINE EVALUATION: OCR vs GPT-4o  ({len(DOSSIERS)} dossiers: {', '.join(d.upper() for d in DOSSIERS)})")
    print("=" * 72)

    agg = {"ocr": [], "gpt4o": []}  # collect per-dossier metrics

    for d in DOSSIERS:
        print(f"\n{'─' * 72}")
        print(f"DOSSIER {d.upper()}")
        print(f"{'─' * 72}")

        gt_bounds, gt_ann = load_gt(d)
        ocr_segs          = load_ocr(d)
        gpt4o_data        = load_gpt4o(d)
        gpt4o_pages       = gpt4o_data["pages"]
        total_pages       = len(gpt4o_pages)

        ocr_bounds  = ocr_boundaries(ocr_segs)
        gpt_bounds  = gpt4o_boundaries(gpt4o_pages)

        print(f"\n  Pages: {total_pages}  |  GT segments: {len(gt_bounds)}")

        # ── Boundary detection ─────────────────────────────────────────────
        print("\n  BOUNDARY DETECTION")
        print(f"  {'Metric':<22} {'OCR':>8} {'GPT-4o':>8}")
        print(f"  {'─'*22} {'─'*8} {'─'*8}")

        for tol_label, tol in [("Exact", 0), ("±1 page", 1)]:
            om = boundary_metrics(ocr_bounds, gt_bounds, tol)
            gm = boundary_metrics(gpt_bounds, gt_bounds, tol)
            print(f"  {tol_label + ' Precision':<22} {om['precision']:>7.1%} {gm['precision']:>7.1%}")
            print(f"  {tol_label + ' Recall':<22} {om['recall']:>7.1%} {gm['recall']:>7.1%}")
            print(f"  {tol_label + ' F1':<22} {om['f1']:>7.1%} {gm['f1']:>7.1%}")
            if tol == 0:
                print(f"  {'GT / OCR pred / GPT pred':<22} {len(gt_bounds):>4}/{len(ocr_bounds):<3} {len(gpt_bounds):>4}/{'-':<2}")
                print()

        agg["ocr"].append(boundary_metrics(ocr_bounds, gt_bounds, 1)["f1"])
        agg["gpt4o"].append(boundary_metrics(gpt_bounds, gt_bounds, 1)["f1"])

        # ── Type classification ────────────────────────────────────────────
        type_results, ocr_tacc, gpt_tacc = type_accuracy(
            gt_bounds, gt_ann, ocr_segs, gpt4o_pages, total_pages
        )

        print(f"\n  TYPE CLASSIFICATION  (per GT segment, using majority vote for GPT-4o)")
        print(f"  {'Metric':<22} {'OCR':>8} {'GPT-4o':>8}")
        print(f"  {'─'*22} {'─'*8} {'─'*8}")
        print(f"  {'Accuracy':<22} {ocr_tacc:>7.1%} {gpt_tacc:>7.1%}")

        # Per-type breakdown
        gt_type_counts = Counter(r["gt"] for r in type_results)
        ocr_correct_by_type  = Counter(r["gt"] for r in type_results if r["ocr_ok"])
        gpt_correct_by_type  = Counter(r["gt"] for r in type_results if r["gpt4o_ok"])

        unique_types = sorted(gt_type_counts)
        if unique_types:
            print(f"\n  Per-type accuracy:")
            print(f"  {'Type':<18} {'Count':>5}  {'OCR':>7}  {'GPT-4o':>7}")
            print(f"  {'─'*18} {'─'*5}  {'─'*7}  {'─'*7}")
            for t in unique_types:
                n = gt_type_counts[t]
                o = ocr_correct_by_type[t] / n
                g = gpt_correct_by_type[t] / n
                print(f"  {t:<18} {n:>5}  {o:>7.1%}  {g:>7.1%}")

        # Mistakes detail
        mistakes_ocr  = [(r["start"], r["gt"], r["ocr"])  for r in type_results if not r["ocr_ok"]]
        mistakes_gpt  = [(r["start"], r["gt"], r["gpt4o"]) for r in type_results if not r["gpt4o_ok"]]
        if mistakes_ocr:
            print(f"\n  OCR misclassifications (pg → gt / pred):")
            for pg, gt, pred in mistakes_ocr[:10]:
                print(f"    pg {pg:3d}: {gt} → {pred}")
        if mistakes_gpt:
            print(f"\n  GPT-4o misclassifications (pg → gt / pred):")
            for pg, gt, pred in mistakes_gpt[:10]:
                print(f"    pg {pg:3d}: {gt} → {pred}")

        # ── Email counts ───────────────────────────────────────────────────
        email_results = email_count_accuracy(
            gt_bounds, gt_ann, ocr_segs, gpt4o_data, total_pages
        )
        if email_results:
            print(f"\n  EMAIL COUNT (for Email-typed segments with known count)")
            print(f"  {'pg':>4}  {'GT':>4}  {'OCR':>6}  {'GPT-4o':>8}  OCR✓  GPT✓")
            print(f"  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*5}")
            gpt_exact = ocr_exact = 0
            for r in email_results:
                ocr_ok = "✓" if r["ocr_count"] == r["gt_count"] else "✗"
                gpt_ok = "✓" if r["gpt4o_count"] == r["gt_count"] else "✗"
                if r["ocr_count"]   == r["gt_count"]: ocr_exact += 1
                if r["gpt4o_count"] == r["gt_count"]: gpt_exact += 1
                print(f"  {r['start']:>4}  {r['gt_count']:>4}  "
                      f"{str(r['ocr_count']):>6}  {str(r['gpt4o_count']):>8}  "
                      f"  {ocr_ok}     {gpt_ok}")
            n = len(email_results)
            print(f"  Exact match: OCR {ocr_exact}/{n} = {ocr_exact/n:.0%}  "
                  f"GPT-4o {gpt_exact}/{n} = {gpt_exact/n:.0%}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"AGGREGATE SUMMARY (across {len(DOSSIERS)} dossiers)")
    print(f"{'=' * 72}")
    print(f"\n  Boundary F1 (±1 page tolerance):")
    print(f"  {'Dossier':<12} {'OCR F1':>8} {'GPT-4o F1':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*10}")
    for i, d in enumerate(DOSSIERS):
        print(f"  {d.upper():<12} {agg['ocr'][i]:>8.1%} {agg['gpt4o'][i]:>10.1%}")
    avg_ocr  = sum(agg["ocr"])   / len(agg["ocr"])
    avg_gpt  = sum(agg["gpt4o"]) / len(agg["gpt4o"])
    print(f"  {'Average':<12} {avg_ocr:>8.1%} {avg_gpt:>10.1%}")
    print()


if __name__ == "__main__":
    run()
