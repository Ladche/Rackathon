#!/usr/bin/env bash
# Container demo: prove the mammography model families actually run end-to-end.
#
# The real sample `data/data.csv` is gitignored and intentionally absent, so this
# demo fits the model families on a clearly-labelled SYNTHETIC sample. If you
# mount a real sample at /app/data/data.csv it will additionally be schema-audited
# (inspect), but note that a sample spanning too few invitation dates cannot
# support chronological benchmarking by design.
set -euo pipefail

PY=python
PIPE=mammography_pipeline.py
OUT=artifacts/docker_demo
SAMPLE=artifacts/synthetic_sample.csv
OUTCOME_END="${OUTCOME_DATA_END:-2026-01-02}"
BUDGET="${BUDGET_SECONDS:-120}"
MODELS="${MODELS:-logistic,decision_tree,random_forest,extra_trees,hist_gradient_boosting,mlp}"
FEATURE_SETS="${FEATURE_SETS:-raw_snapshot,snapshot_engineered}"

echo "=================================================================="
echo " Mammography pipeline — Docker demo"
echo " Python: $($PY --version 2>&1)"
echo "=================================================================="

echo
echo ">>> [1/6] Fast self-test (synthetic diagnostics, no data files)"
$PY "$PIPE" self-test

if [ -f data/data.csv ]; then
  echo
  echo ">>> [1b] Real sample detected at data/data.csv — schema audit only"
  $PY "$PIPE" inspect --data data/data.csv --data-mode sample \
    --output-dir "$OUT/real_sample_audit" || \
    echo "    (inspect returned non-zero; see output above)"
else
  echo
  echo ">>> Note: data/data.csv is absent (expected). Using synthetic data for the model run."
fi

echo
echo ">>> [2/6] Generate SYNTHETIC sample"
$PY docker/make_synthetic_sample.py --output "$SAMPLE"

echo
echo ">>> [3/6] Benchmark — fit all requested model families on validation folds"
$PY "$PIPE" benchmark \
  --data "$SAMPLE" --data-mode sample \
  --date-format "%Y-%m-%d" --assume-t0-snapshots \
  --outcome-data-end "$OUTCOME_END" --test-year 2024 \
  --models "$MODELS" --feature-sets "$FEATURE_SETS" \
  --targets 90,180,365 --preset quick --budget-seconds "$BUDGET" --n-jobs 2 \
  --save-model --output-dir "$OUT/run" \
  --provenance "SYNTHETIC mechanics demo inside Docker; not empirical evidence"

echo
echo ">>> [4/6] Evaluate the frozen bundle on the locked synthetic test split"
$PY "$PIPE" evaluate \
  --model-path "$OUT/run/model_bundle.joblib" \
  --data "$SAMPLE" --data-mode sample --split test \
  --output-dir "$OUT/run/final_evaluation"

echo
echo ">>> [5/6] Predict — score shuffled, target-free invitations (fresh process)"
$PY - "$SAMPLE" <<'PYEOF'
import sys, pandas as pd
src = sys.argv[1]
df = pd.read_csv(src, dtype=str)
df = df[df["invitation_date"].str.startswith("2024")].drop(
    columns=["screening_90d", "screening_180d", "screening_365d"])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv("artifacts/predict_input.csv", index=False)
print(f"    Prepared {len(df)} target-free invitations (shuffled).")
PYEOF
$PY "$PIPE" predict \
  --model-path "$OUT/run/model_bundle.joblib" \
  --data artifacts/predict_input.csv --data-mode sample \
  --data-as-of "$OUTCOME_END" \
  --output "$OUT/predictions_demo.csv"

echo
echo ">>> [6/6] Summary"
echo "    Validation leaderboard (top rows):"
head -n 6 "$OUT/run/validation_leaderboard.csv" || true
echo
echo "    Example coherent predictions (first rows):"
head -n 4 "$OUT/predictions_demo.csv" || true
echo
echo "=================================================================="
echo " Demo complete. Artifacts under: $OUT"
echo "=================================================================="
