"""
Pipeline evaluation: OCR vs GPT-4o across annotated dossiers.
Compares boundary detection and document type classification.

Auto-detects dossiers by scanning groundtruth/annotations_dossier_X.json files
(the rich format auto-saved by annotate.py). Skips dossiers missing either the
OCR segs or GPT-4o cache file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION METRICS (following van Heusden et al., OpenPSS 2024)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PAGE-LEVEL METRICS
------------------
Each page is treated as a binary classification: does this page start a new
document (1) or is it a continuation (0)?

  Precision  = TP / (TP + FP)
             = fraction of predicted boundaries that are correct

  Recall     = TP / (TP + FN)
             = fraction of true boundaries that were found

  F1         = 2 * P * R / (P + R)
             = harmonic mean of precision and recall

  With tolerance ±1: a predicted boundary within 1 page of a GT boundary
  counts as a TP (useful for near-miss cases).

  Note: page-level F1 treats all boundaries equally regardless of how long
  the affected documents are — a missed boundary in a 10-page doc counts the
  same as one in a 1-page doc.

DOCUMENT-LEVEL METRICS  (Panoptic Quality framework, Kirillov et al. 2019)
---------------------------------------------------------------------------
Each document is treated as a SET of pages. Predicted and GT documents are
matched by their Intersection over Union (IoU):

  IoU(t, p)  = |pages(t) ∩ pages(p)| / |pages(t) ∪ pages(p)|

A pair (t, p) is a TRUE POSITIVE if IoU(t, p) > 0.5. This threshold enforces
that each true or predicted document can be part of at most one TP pair —
a document cannot simultaneously be a good match for two different predictions.

  TP  = set of matched (gt_doc, pred_doc) pairs with IoU > 0.5
  FP  = predicted documents not matched to any GT doc
  FN  = GT documents not matched to any predicted doc

  Recognition Quality (RQ) = unweighted document F1
      Precision_doc = |TP| / (|TP| + |FP|)
      Recall_doc    = |TP| / (|TP| + |FN|)
      RQ            = 2 * P_doc * R_doc / (P_doc + R_doc)

  Segmentation Quality (SQ) = mean IoU of all TP pairs
      SQ = sum(IoU(t,p) for (t,p) in TP) / |TP|
      Measures how precisely the boundaries of correctly identified documents
      are placed — 1.0 means perfect boundary alignment for all matched docs.

  Panoptic Quality (PQ) = weighted document F1
      PQ = RQ * SQ
      Equivalent to: sum(IoU(t,p) for TP) / (|TP| + 0.5*|FP| + 0.5*|FN|)
      Penalises both wrong splits/merges (via RQ) and imprecise boundaries
      (via SQ). This is the headline metric from the OpenPSS paper.

TYPE CLASSIFICATION
-------------------
For each GT document segment, the majority predicted category across its pages
is compared to the ground-truth label. Accuracy = fraction of GT segments
where the predicted type matches exactly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
from collections import Counter
from pathlib import Path

GT_DIR = Path("groundtruth")


def _discover_dossiers() -> list[str]:
    """Return sorted list of dossier letters that have all three required files."""
    letters = []
    for ann in sorted(GT_DIR.glob("annotations_dossier_?.json")):
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
    """Load ground truth from groundtruth/annotations_dossier_X.json.

    Handles two formats:
    - Rich format (annotate.py auto-save): {documents: [{start_page, type_gt, ...}], ...}
    - Simple format (older export):        {boundaries: [...], annotations: {...}}
    """
    path = GT_DIR / f"annotations_dossier_{dossier}.json"
    with open(path) as f:
        data = json.load(f)

    if "documents" in data:
        docs = data["documents"]
        boundaries = [doc["start_page"] for doc in docs]
        annotations = {
            str(doc["start_page"]): {
                "type_gt":       doc.get("type_gt", ""),
                "num_emails_gt": doc.get("num_emails_gt"),
                "emails_gt":     doc.get("emails_gt", []),
                "notes":         doc.get("notes", ""),
            }
            for doc in docs
        }
    else:
        boundaries  = data["boundaries"]
        annotations = data["annotations"]

    return boundaries, annotations


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


_NOTA_LIKE = frozenset({"Nota", "Brief", "Report", "Vergadernotulen"})


def gpt4o_boundaries(gpt4o_data: dict) -> list[int]:
    """
    Return sorted 0-indexed start pages using the most authoritative source:

    - boundary_documents (pass-2 LLM) for unstamped dossiers
    - Forward-fill boundary logic for stamped dossiers — mirrors
      _build_docs_forward_fill: a boundary is where the stamp code changes,
      OR where within_doc_page==1 reoccurs on a nota-like doc within the same
      stamp code (sub-split). This is what the pipeline actually produces.
    - is_new_document flags as a last-resort fallback.
    """
    pages = gpt4o_data.get("pages", [])
    if not pages:
        return []

    bd = gpt4o_data.get("boundary_documents", [])
    if bd:
        # Unstamped: boundary_documents is the pass-2 LLM result (1-based start_page)
        starts = {b["start_page"] - 1 for b in bd if b.get("start_page")}
        starts.add(0)
        return sorted(starts)

    # Stamped: replicate _build_docs_forward_fill boundary detection
    last_detected: str | None = None
    boundaries: set[int] = {0}
    for p in pages:
        detected = p.get("doc_code")
        wpn      = p.get("within_doc_page")
        category = p.get("category", "Other")
        is_new   = (p.get("is_new_document", False) or wpn == 1) and (wpn is None or wpn == 1)
        pg0      = p["page_num"] - 1

        if detected and detected != last_detected:
            boundaries.add(pg0)
            last_detected = detected
        elif detected and is_new and category in _NOTA_LIKE:
            boundaries.add(pg0)
            last_detected = detected
        elif detected:
            last_detected = detected
        elif is_new and last_detected is not None:
            boundaries.add(pg0)
            last_detected = None

    if len(boundaries) > 1:
        return sorted(boundaries)

    # Fallback: is_new_document flags (pass-1 vision, rarely reliable)
    for p in pages:
        if p.get("is_new_document"):
            boundaries.add(p["page_num"] - 1)
    return sorted(boundaries)


def _boundaries_to_segments(boundaries: list[int], total_pages: int) -> list[set[int]]:
    """
    Convert a sorted list of 0-indexed boundary start pages into a list of
    page sets, one per document.

    Example: boundaries=[0, 3, 7], total_pages=10
      → [{0,1,2}, {3,4,5,6}, {7,8,9}]
    """
    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else total_pages
        segments.append(set(range(start, end)))
    return segments


def page_level_metrics(predicted: list[int], ground_truth: list[int],
                       total_pages: int, tolerance: int = 0) -> dict:
    """
    Compute page-level precision, recall and F1 for boundary detection.

    Each page is a binary label: 1 if it starts a new document, 0 otherwise.
    With tolerance > 0, a predicted boundary within ±tolerance pages of a GT
    boundary counts as a TP (matched greedily, each GT boundary at most once).
    """
    pred_set = set(predicted)
    gt_set   = set(ground_truth)

    if tolerance == 0:
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
    else:
        matched_gt   = set()
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


# Keep old name as alias used in aggregate collection
boundary_metrics = page_level_metrics


def panoptic_quality(pred_boundaries: list[int], gt_boundaries: list[int],
                     total_pages: int) -> dict:
    """
    Compute document-level metrics using the Panoptic Quality framework
    (Kirillov et al., CVPR 2019; used in van Heusden et al., OpenPSS 2024).

    Steps:
      1. Convert boundary lists to sets-of-pages (one set per document).
      2. For each (gt_doc, pred_doc) pair, compute IoU = |intersection| / |union|.
      3. A pair is a TP if IoU > 0.5. Each doc appears in at most one TP pair
         (the IoU > 0.5 threshold mathematically guarantees this).
      4. Unmatched predicted docs → FP; unmatched GT docs → FN.

    Returns:
      precision_doc  : |TP| / (|TP| + |FP|)
      recall_doc     : |TP| / (|TP| + |FN|)
      rq             : Recognition Quality — unweighted document F1
      sq             : Segmentation Quality — mean IoU of TP pairs (boundary precision)
      pq             : Panoptic Quality = RQ * SQ (weighted document F1)
      n_tp, n_fp, n_fn
    """
    gt_segs   = _boundaries_to_segments(gt_boundaries,   total_pages)
    pred_segs = _boundaries_to_segments(pred_boundaries, total_pages)

    matched_gt   = set()
    matched_pred = set()
    iou_sum      = 0.0

    for gi, g in enumerate(gt_segs):
        for pi, p in enumerate(pred_segs):
            if pi in matched_pred:
                continue
            intersection = len(g & p)
            if intersection == 0:
                continue
            union = len(g | p)
            iou   = intersection / union
            if iou > 0.5:
                matched_gt.add(gi)
                matched_pred.add(pi)
                iou_sum += iou
                break   # each GT doc matches at most one pred doc

    n_tp = len(matched_gt)
    n_fp = len(pred_segs) - len(matched_pred)
    n_fn = len(gt_segs)   - len(matched_gt)

    prec_doc = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    rec_doc  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    rq       = 2 * prec_doc * rec_doc / (prec_doc + rec_doc) if (prec_doc + rec_doc) > 0 else 0.0
    sq       = iou_sum / n_tp if n_tp > 0 else 0.0
    pq       = rq * sq

    return {
        "precision_doc": prec_doc,
        "recall_doc":    rec_doc,
        "rq":            rq,
        "sq":            sq,
        "pq":            pq,
        "n_tp":          n_tp,
        "n_fp":          n_fp,
        "n_fn":          n_fn,
    }


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
            "start":    start,
            "end":      end,
            "gt":       gt_type,
            "ocr":      ocr_type,
            "gpt4o":    gpt_type,
            "ocr_ok":   ocr_type == gt_type,
            "gpt4o_ok": gpt_type == gt_type,
        })
    ocr_acc   = sum(r["ocr_ok"]   for r in results) / len(results) if results else 0
    gpt4o_acc = sum(r["gpt4o_ok"] for r in results) / len(results) if results else 0
    return results, ocr_acc, gpt4o_acc


def _resolve_gpt4o_code(start_0indexed: int, gpt4o_data: dict) -> str | None:
    """
    Find the final pipeline doc code the GPT-4o pipeline assigned to the page at
    start_0indexed. The source depends on how the pipeline built its docs:

    1. boundary_documents (pass-2 LLM) — present for unstamped dossiers.
    2. Per-page doc_code — fallback for stamped dossiers using forward-fill.
    """
    import re as _re

    bd = gpt4o_data.get("boundary_documents", [])
    if bd:
        docs_sorted = sorted(bd, key=lambda d: d.get("start_page", 1))
        containing_idx = None
        for i, doc in enumerate(docs_sorted):
            if doc.get("start_page", 1) - 1 <= start_0indexed:
                containing_idx = i
            else:
                break
        if containing_idx is None:
            return None
        raw_code = docs_sorted[containing_idx].get("doc_code")
        if raw_code:
            s = _re.sub(r"\D", "", str(raw_code))
            if len(s) == 4 and _re.fullmatch(r"0\d{3}", s) and not _re.fullmatch(r"(19|20)\d{2}", s):
                return s
        return f"auto_{containing_idx + 1:03d}"

    pages = gpt4o_data.get("pages", [])
    first_page = next((p for p in pages if p["page_num"] - 1 == start_0indexed), None)
    if first_page and first_page.get("doc_code"):
        return first_page["doc_code"]
    return None


def email_count_accuracy(gt_boundaries, gt_annotations, ocr_segments, gpt4o_data, total_pages):
    """Compare email sub-document counts for Email-typed GT segments."""
    results = []
    emails_by_doc = gpt4o_data.get("emails_by_doc", {})

    for i, start in enumerate(gt_boundaries):
        end = gt_boundaries[i + 1] - 1 if i + 1 < len(gt_boundaries) else total_pages - 1
        ann = gt_annotations.get(str(start), {})
        if ann.get("type_gt") != "E-mail":
            continue
        gt_count = ann.get("num_emails_gt")
        if gt_count is None:
            continue

        ocr_emails = None
        for seg in ocr_segments:
            if seg["start_page"] <= start <= seg["end_page"]:
                ocr_emails = seg.get("num_emails", None)
                break

        code = _resolve_gpt4o_code(start, gpt4o_data)
        gpt_emails = None
        if code:
            entries = emails_by_doc.get(code, [])
            gpt_emails = len(entries) if entries else None

        results.append({
            "start":      start,
            "gt_count":   gt_count,
            "ocr_count":  ocr_emails,
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

    # Aggregate collectors
    agg = {
        "ocr":   {"page_p": [], "page_r": [], "page_f1": [], "rq": [], "sq": [], "pq": [], "type_acc": []},
        "gpt4o": {"page_p": [], "page_r": [], "page_f1": [], "rq": [], "sq": [], "pq": [], "type_acc": []},
    }

    for d in DOSSIERS:
        print(f"\n{'─' * 72}")
        print(f"DOSSIER {d.upper()}")
        print(f"{'─' * 72}")

        gt_bounds, gt_ann = load_gt(d)
        ocr_segs          = load_ocr(d)
        gpt4o_data        = load_gpt4o(d)
        gpt4o_pages       = gpt4o_data["pages"]
        total_pages       = len(gpt4o_pages)

        ocr_bounds = ocr_boundaries(ocr_segs)
        gpt_bounds = gpt4o_boundaries(gpt4o_data)

        print(f"\n  Pages: {total_pages}  |  GT segments: {len(gt_bounds)}  |  "
              f"OCR predicted: {len(ocr_bounds)}  |  GPT-4o predicted: {len(gpt_bounds)}")

        # ── Page-level metrics ─────────────────────────────────────────────
        print("\n  PAGE-LEVEL METRICS")
        print(f"  {'Metric':<26} {'OCR':>8} {'GPT-4o':>8}")
        print(f"  {'─'*26} {'─'*8} {'─'*8}")

        for tol_label, tol in [("Exact", 0), ("±1 page", 1)]:
            om = page_level_metrics(ocr_bounds, gt_bounds, total_pages, tol)
            gm = page_level_metrics(gpt_bounds, gt_bounds, total_pages, tol)
            print(f"  {tol_label + ' Precision':<26} {om['precision']:>7.1%} {gm['precision']:>7.1%}")
            print(f"  {tol_label + ' Recall':<26} {om['recall']:>7.1%} {gm['recall']:>7.1%}")
            print(f"  {tol_label + ' F1':<26} {om['f1']:>7.1%} {gm['f1']:>7.1%}")
            if tol == 0:
                print()

        # ── Document-level metrics (Panoptic Quality) ──────────────────────
        o_pq = panoptic_quality(ocr_bounds, gt_bounds, total_pages)
        g_pq = panoptic_quality(gpt_bounds, gt_bounds, total_pages)

        print(f"\n  DOCUMENT-LEVEL METRICS  (IoU > 0.5 matching)")
        print(f"  {'Metric':<26} {'OCR':>8} {'GPT-4o':>8}")
        print(f"  {'─'*26} {'─'*8} {'─'*8}")
        print(f"  {'Doc Precision':<26} {o_pq['precision_doc']:>7.1%} {g_pq['precision_doc']:>7.1%}")
        print(f"  {'Doc Recall':<26} {o_pq['recall_doc']:>7.1%} {g_pq['recall_doc']:>7.1%}")
        print(f"  {'RQ  (Doc F1 unweighted)':<26} {o_pq['rq']:>7.1%} {g_pq['rq']:>7.1%}")
        print(f"  {'SQ  (mean IoU of TPs)':<26} {o_pq['sq']:>7.3f} {g_pq['sq']:>7.3f}")
        print(f"  {'PQ  (Doc F1 weighted)':<26} {o_pq['pq']:>7.1%} {g_pq['pq']:>7.1%}")
        print(f"  {'TP / FP / FN':<26} "
              f"  {o_pq['n_tp']}/{o_pq['n_fp']}/{o_pq['n_fn']}    "
              f"  {g_pq['n_tp']}/{g_pq['n_fp']}/{g_pq['n_fn']}")

        # ── Type classification ────────────────────────────────────────────
        type_results, ocr_tacc, gpt_tacc = type_accuracy(
            gt_bounds, gt_ann, ocr_segs, gpt4o_pages, total_pages
        )

        # Collect for aggregate
        om1 = page_level_metrics(ocr_bounds, gt_bounds, total_pages, 1)
        gm1 = page_level_metrics(gpt_bounds, gt_bounds, total_pages, 1)
        agg["ocr"]["page_p"].append(om1["precision"])
        agg["ocr"]["page_r"].append(om1["recall"])
        agg["ocr"]["page_f1"].append(om1["f1"])
        agg["ocr"]["rq"].append(o_pq["rq"])
        agg["ocr"]["sq"].append(o_pq["sq"])
        agg["ocr"]["pq"].append(o_pq["pq"])
        agg["ocr"]["type_acc"].append(ocr_tacc)
        agg["gpt4o"]["page_p"].append(gm1["precision"])
        agg["gpt4o"]["page_r"].append(gm1["recall"])
        agg["gpt4o"]["page_f1"].append(gm1["f1"])
        agg["gpt4o"]["rq"].append(g_pq["rq"])
        agg["gpt4o"]["sq"].append(g_pq["sq"])
        agg["gpt4o"]["pq"].append(g_pq["pq"])
        agg["gpt4o"]["type_acc"].append(gpt_tacc)

        print(f"\n  TYPE CLASSIFICATION  (majority vote per GT segment)")
        print(f"  {'Metric':<26} {'OCR':>8} {'GPT-4o':>8}")
        print(f"  {'─'*26} {'─'*8} {'─'*8}")
        print(f"  {'Accuracy':<26} {ocr_tacc:>7.1%} {gpt_tacc:>7.1%}")

        gt_type_counts      = Counter(r["gt"] for r in type_results)
        ocr_correct_by_type = Counter(r["gt"] for r in type_results if r["ocr_ok"])
        gpt_correct_by_type = Counter(r["gt"] for r in type_results if r["gpt4o_ok"])

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

        mistakes_ocr = [(r["start"], r["gt"], r["ocr"])   for r in type_results if not r["ocr_ok"]]
        mistakes_gpt = [(r["start"], r["gt"], r["gpt4o"]) for r in type_results if not r["gpt4o_ok"]]
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

    # ── Aggregate summary — academic table (mirrors OpenPSS Table 2) ─────────
    avg = lambda lst: sum(lst) / len(lst)

    print(f"\n{'=' * 96}")
    print(f"AGGREGATE SUMMARY (across {len(DOSSIERS)} dossiers)")
    print(f"{'=' * 96}")
    print()
    print("  Metrics follow van Heusden et al. (OpenPSS 2024):")
    print("  Page P/R/F1: page-level boundary detection (±1 tolerance).")
    print("  RQ: Recognition Quality = unweighted doc F1 (IoU > 0.5 matching).")
    print("  SQ: Segmentation Quality = mean IoU of matched doc pairs.")
    print("  PQ: Panoptic Quality = RQ × SQ (weighted doc F1).")
    print("  Type Acc: fraction of GT segments with correct type prediction.")
    print()

    # Header — two pipelines side by side, matching paper column order
    col_w = 6
    def pct(v): return f"{v:.3f}"
    def flt(v): return f"{v:.3f}"

    # Two parallel tables — one per pipeline — each with full Page + Doc + Type columns
    hdr = (f"  {'Dossier':<12}  {'P':>{col_w}}  {'R':>{col_w}}  {'F1':>{col_w}}  "
           f"{'RQ':>{col_w}}  {'SQ':>{col_w}}  {'PQ':>{col_w}}  {'Type Acc':>{col_w+2}}")
    div = "  " + "─" * (len(hdr) - 2)

    def _row(d_label, pp, pr, pf, rq, sq, pq, ta):
        return (f"  {d_label:<12}  {pct(pp):>{col_w}}  {pct(pr):>{col_w}}  {pct(pf):>{col_w}}  "
                f"{pct(rq):>{col_w}}  {flt(sq):>{col_w}}  {pct(pq):>{col_w}}  {pct(ta):>{col_w+2}}")

    for pipeline in ("ocr", "gpt4o"):
        label = "OCR pipeline" if pipeline == "ocr" else "GPT-4o pipeline"
        a = agg[pipeline]
        print(f"  {label}")
        print(f"  {'':12}  {'── Page (±1 tol.) ──':^22}  {'── Document (IoU>0.5) ───':^24}  {'':>8}")
        print(hdr)
        print(div)
        for i, d in enumerate(DOSSIERS):
            print(_row(d.upper(),
                       a["page_p"][i], a["page_r"][i], a["page_f1"][i],
                       a["rq"][i], a["sq"][i], a["pq"][i], a["type_acc"][i]))
        print(div)
        print(_row("Average",
                   avg(a["page_p"]), avg(a["page_r"]), avg(a["page_f1"]),
                   avg(a["rq"]), avg(a["sq"]), avg(a["pq"]), avg(a["type_acc"])))
        print()


if __name__ == "__main__":
    run()
