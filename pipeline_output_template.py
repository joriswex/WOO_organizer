"""
pipeline_output_template.py
----------------------------
Generates a prediction template from a gold standard file, and shows
exactly what format your pipeline must produce.

Run this first to understand what your pipeline needs to output, then
fill in the predictions and run evaluate_pss.py.

USAGE
-----
    python pipeline_output_template.py \
        --gold gold_standard_sample.json \
        --out  my_predictions.json

This writes a predictions file pre-filled with all zeros (no boundaries),
which you replace with your pipeline's actual predictions.

═══════════════════════════════════════════════════════════════
  REQUIRED OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

A single JSON file, structured identically to the gold standard:

    {
      "stream_16": [1, 0, 0, 0, 0, 0, ...],   ← list length = n_pages in stream
      "stream_35": [1, 0, 1, 0, 1, ...],
      ...
    }

Rules:
  1. Keys must match gold standard stream names exactly.
  2. Each list must have EXACTLY the same length as the gold standard list.
  3. Values must be 0 or 1 (or floats ≥0.5 treated as 1 by the evaluator).
  4. The first element of every list MUST be 1 — it represents the trivial
     fact that the first page of a stream starts a document. The evaluator
     excludes it from scoring, but it must be present to keep indices aligned.

═══════════════════════════════════════════════════════════════
  HOW YOUR PIPELINE SHOULD PRODUCE PREDICTIONS
═══════════════════════════════════════════════════════════════

For each stream (i.e. each PDF in the dataset):

    pages = load_stream_pages(stream_name)  # list of page images / OCR text

    predictions = [1]  # first page always starts a document

    for i in range(1, len(pages)):
        # Your model sees page i-1 and page i (or just page i)
        # and decides: does page i start a new document?
        label = your_model.predict(pages[i-1], pages[i])  # returns 0 or 1
        predictions.append(label)

    output[stream_name] = predictions

═══════════════════════════════════════════════════════════════
  CONNECTING TO THE GOLD STANDARD FILE STRUCTURE
═══════════════════════════════════════════════════════════════

The gold standard JSON (gold_standard_sample.json) maps each stream name
to the list of ground-truth labels. The stream name is both the key in
the JSON and the name of the corresponding PDF/folder in the dataset.

Example: stream "stream_35" has 1103 pages.
  gold_standard["stream_35"]  → list of 1103 labels
  your_predictions["stream_35"] → list of 1103 predictions

Page index 0 → always 1 (excluded from evaluation)
Page index 1 → first real transition evaluated
...
Page index 1102 → last transition evaluated

═══════════════════════════════════════════════════════════════
  QUICK SANITY CHECKS BEFORE RUNNING EVALUATE_PSS.PY
═══════════════════════════════════════════════════════════════

    python pipeline_output_template.py --validate \
        --gold gold_standard_sample.json \
        --pred my_predictions.json
"""

import json
import argparse
import sys


def generate_template(gold_path: str, out_path: str):
    with open(gold_path) as f:
        gold = json.load(f)

    template = {}
    for stream, labels in gold.items():
        # Pre-fill with all-zero predictions (predict no boundaries)
        # except the mandatory first-page 1
        template[stream] = [1] + [0] * (len(labels) - 1)

    with open(out_path, "w") as f:
        json.dump(template, f, indent=2)

    total_pages = sum(len(v) for v in template.values())
    print(f"Template written to: {out_path}")
    print(f"  {len(template)} streams, {total_pages:,} pages total")
    print(f"\nReplace the 0s with your pipeline's predictions, then run:")
    print(f"  python evaluate_pss.py --gold {gold_path} --pred {out_path} --name 'My Pipeline'")


def validate(gold_path: str, pred_path: str):
    with open(gold_path) as f:
        gold = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)

    errors = []
    warnings = []

    for stream in gold:
        if stream not in pred:
            errors.append(f"Stream '{stream}' missing from predictions.")
            continue
        if len(pred[stream]) != len(gold[stream]):
            errors.append(
                f"Stream '{stream}': gold has {len(gold[stream])} pages, "
                f"pred has {len(pred[stream])}."
            )
        if pred[stream][0] != 1:
            warnings.append(
                f"Stream '{stream}': first prediction is {pred[stream][0]}, "
                f"expected 1 (will be ignored in evaluation anyway)."
            )
        invalid = [v for v in pred[stream] if v not in (0, 1) and not (0.0 <= v <= 1.0)]
        if invalid:
            errors.append(
                f"Stream '{stream}': {len(invalid)} values outside [0, 1]: "
                f"{invalid[:5]}{'...' if len(invalid) > 5 else ''}"
            )

    for s in pred:
        if s not in gold:
            warnings.append(f"Stream '{s}' in pred but not in gold (will be ignored).")

    if errors:
        print("ERRORS (must fix before evaluating):")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not errors and not warnings:
        print(f"✓ Predictions file looks valid! {len(pred)} streams, "
              f"{sum(len(v) for v in pred.values()):,} pages.")
        print("  Ready to run: python evaluate_pss.py --gold ... --pred ...")
    elif not errors:
        print("\n✓ No errors found (warnings are non-critical). Ready to evaluate.")
    else:
        print(f"\n✗ {len(errors)} error(s) found. Fix these before evaluating.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold",     required=True)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--pred",     default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        if not args.pred:
            print("--pred required for --validate")
            sys.exit(1)
        validate(args.gold, args.pred)
    else:
        if not args.out:
            print("--out required to generate template")
            sys.exit(1)
        generate_template(args.gold, args.out)
