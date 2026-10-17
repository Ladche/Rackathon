#!/usr/bin/env bash
# Run the full pipeline on the REAL local CSV (data/Rakhaton2026.csv) inside the container.
#
# Patient-level data stays local: the CSV is mounted read-only and every output goes to
# the gitignored ./artifacts directory. Nothing is uploaded anywhere.
#
# OUTCOME_DATA_END is NOT given by the dictionary. The default below is a clearly
# labelled WORKING ASSUMPTION (last invitation in the file is early 2024, so a
# 365-day horizon is complete by end of 2025). Override it with a verified value.
set -euo pipefail

PY=python
PIPE=mammography_pipeline.py
DATA="${DATA_FILE:-data/Rakhaton2026.csv}"
OUT="${OUT_DIR:-artifacts/real_run}"
OUTCOME_END="${OUTCOME_DATA_END:-2026-01-02}"
LAG="${REPORTING_LAG_DAYS:-0}"
TEST_YEAR="${TEST_YEAR:-2024}"
MEM="${MEMORY_MB:-14000}"
BUDGET="${BUDGET_SECONDS:-600}"
PRESET="${PRESET:-quick}"
NJOBS="${N_JOBS:-2}"
MODELS="${MODELS:-logistic,decision_tree,random_forest,extra_trees,hist_gradient_boosting,mlp}"
FEATURE_SETS="${FEATURE_SETS:-raw_snapshot,snapshot_engineered}"
PEOPLE="${PATIENT_SAMPLE:-}"

echo "=================================================================="
echo " Mammography pipeline - REAL DATA run"
echo " data=$DATA  outcome_end=$OUTCOME_END (assumption unless verified)"
echo " lag=$LAG test_year=$TEST_YEAR memory_mb=$MEM patient_sample=${PEOPLE:-all}"
echo " models=$MODELS"
echo "=================================================================="

if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA not found. Mount it, e.g. -v ./data:/app/data:ro" >&2
  exit 3
fi

mkdir -p "$OUT"
if [ -n "$PEOPLE" ]; then
  echo; echo ">>> [0/5] streamed whole-patient subsample (bounds RAM)"
  $PY docker/subsample_patients.py --input "$DATA" --output "$OUT/subsample.csv" --people "$PEOPLE"
  DATA="$OUT/subsample.csv"
fi
COMMON=(--data "$DATA" --data-mode full --allow-full-data --memory-mb "$MEM" --chunk-size 100000)

echo; echo ">>> [1/5] inspect (schema audit, no fitting)"
$PY "$PIPE" inspect "${COMMON[@]}" --output-dir "$OUT/audit"

echo; echo ">>> [2/5] benchmark (chronological search, frozen refit, --save-model)"
$PY "$PIPE" benchmark "${COMMON[@]}" \
  --assume-t0-snapshots --outcome-data-end "$OUTCOME_END" --reporting-lag-days "$LAG" \
  --test-year "$TEST_YEAR" --models "$MODELS" --feature-sets "$FEATURE_SETS" \
  --targets 90,180,365 --preset "$PRESET" --budget-seconds "$BUDGET" --n-jobs "$NJOBS" \
  --bootstrap "${BOOTSTRAP:-30}" --save-model --output-dir "$OUT/run" \
  --provenance "REAL VZP extract; outcome end $OUTCOME_END is an ASSUMPTION, not verified"

echo; echo ">>> [3/5] evaluate frozen bundle on the reserved test year"
$PY "$PIPE" evaluate "${COMMON[@]}" \
  --model-path "$OUT/run/model_bundle.joblib" --split test \
  --bootstrap "${BOOTSTRAP:-30}" --output-dir "$OUT/run/final_evaluation"

echo; echo ">>> [4/5] predict on target-free test-year invitations (fresh process)"
$PY - "$DATA" "$TEST_YEAR" "$OUT/predict_input.csv" <<'PYEOF'
import sys, pandas as pd
src, year, dst = sys.argv[1], sys.argv[2], sys.argv[3]
cols = pd.read_csv(src, nrows=0).columns
keep = [c for c in cols if not c.upper().startswith("SCREENING_")]
parts = []
for chunk in pd.read_csv(src, dtype=str, keep_default_na=False, usecols=keep, chunksize=200000):
    parts.append(chunk[chunk["INVITATION_DATE"].str.endswith(year)])
df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv(dst, index=False)
print(f"    Prepared {len(df)} target-free invitations (shuffled) -> {dst}")
PYEOF
$PY "$PIPE" predict \
  --model-path "$OUT/run/model_bundle.joblib" \
  --data "$OUT/predict_input.csv" --data-mode full --allow-full-data --memory-mb "$MEM" \
  --data-as-of "$OUTCOME_END" --output "$OUT/predictions.csv"

echo; echo ">>> [5/5] summary"
echo "    Validation leaderboard (top rows):"
head -n 8 "$OUT/run/validation_leaderboard.csv" || true
echo; echo "    First predictions:"
head -n 4 "$OUT/predictions.csv" || true
echo "=================================================================="
echo " Real-data run complete. Artifacts under: $OUT (gitignored)"
echo "=================================================================="
