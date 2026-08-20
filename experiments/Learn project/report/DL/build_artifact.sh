#!/usr/bin/env bash
# Build the ANONYMIZED supplemental artifact promised in the paper's Data and Code
# Availability statement. ML4H desk-rejects non-anonymized code, so this builds a clean
# tree from scratch rather than zipping the working directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/artifact"
rm -rf "$OUT"; mkdir -p "$OUT"/{code,scores,tables}

# --- code: the single self-contained notebook source, with the project name neutralised
sed -e "s/WANDB_PROJECT  = 'MedIAnomaly-DL'/WANDB_PROJECT  = os.environ.get('WANDB_PROJECT', 'anon-project')/" \
    "$HERE/../../DL/dl_project.py" > "$OUT/code/dl_project.py"
cp "$HERE/verify_paper_numbers.py" "$HERE/ssim_window_sweep.py" \
   "$HERE/sigma_crossing.py" "$HERE/support_decomposition.py" "$OUT/code/"

# --- per-image score vectors + manifests, per dataset. NO weights (size), NO raw logs.
for B in dl_results_RSNA_dl-v1_20260818-1013 \
         dl_results_VinCXR_dl-v1-vincxr_20260818-1047 \
         dl_results_LAG_dl-v1-lag_20260818-1111; do
  DS=$(echo "$B" | sed 's/^dl_results_//; s/_dl-v1.*//')
  for RD in "$HERE/$B/results_dl"/ckpt_*/*/; do
    RID=$(basename "$RD"); D="$OUT/scores/$DS/$RID"; mkdir -p "$D"
    cp "$RD/scores.npy" "$D/" 2>/dev/null || true
    cp "$RD/labels.npy" "$D/" 2>/dev/null || true
    # manifests carry absolute paths and timestamps; keep only the research fields
    python3 - "$RD/manifest.json" "$D/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
json.dump({k:m[k] for k in ('run_id','method','seed','params','metrics','config') if k in m},
          open(sys.argv[2],'w'), indent=2)
PY
  done
  cp "$HERE/$B/results_dl"/*.csv "$OUT/tables/" 2>/dev/null || true
  cp "$HERE/$B/results_dl"/selection_card.json "$OUT/tables/" 2>/dev/null || true
done
# ALLOWLIST. Working narrative documents (07_*, 10_*, 12_*, RESULTS_FINAL, STATUS, the
# audit files) must never ship: they contain superseded framings and retracted claims, and
# a reader who finds them concludes the corrections were cosmetic.
for F in FACTORIAL_GRID.txt WIDTH_SWEEP.txt CLINICAL_OPERATING_POINTS.txt \
         SUPPORT_DECOMPOSITION.txt M3_CORRECTED.txt DARK_REGION_CHECK.txt \
         ssim_window_sweep.json ssim_bandwidth_sweep.json sigma_crossing.json \
         recon_range.json dark_region_check.json support_decomposition.json \
         objective_axis_delong.json m0_ladder.json; do
  [ -f "$HERE/$F" ] && cp "$HERE/$F" "$OUT/tables/"
done

cat > "$OUT/README.md" <<'MDEOF'
# Supplemental artifact (anonymized for review)

Per-image anomaly score vectors for every cell reported in the paper, the run manifests
that produced them, the derived tables, and the code.

    code/dl_project.py          the full experiment, one self-contained file
    code/verify_paper_numbers.py  regenerates every number in the paper and diffs it
    code/ssim_window_sweep.py   the fixed-resolution SSIM window analysis
    code/sigma_crossing.py      the SSIM bandwidth sweep behind Limitation 1
    code/support_decomposition.py  rebuilds the distance/support split and checks it
    scores/<dataset>/<run_id>/  scores.npy, labels.npy, manifest.json
    tables/                     derived CSVs, the selection card, the analysis outputs

Every table in the paper recomputes from `scores/` alone. No GPU and no retraining are
required to place a new anomaly score in the grid: load a run's `scores.npy`, or apply a
new scoring function to a reconstruction and evaluate against `labels.npy`.

`tables/selection_card.json` records the two choices fixed on RSNA (post-hoc head width and
ensemble membership) that were then applied unchanged to VinDr-CXR and LAG. It is what
makes those two datasets held out.

Model weights are omitted for size; they are available on request and are not needed to
reproduce any reported number.
MDEOF

echo "=== ANONYMITY CHECK ==="
FAIL=0
set +e
# identity, and then claims the paper has retracted -- a released file asserting the
# opposite of the paper is worse than any defect in the paper
for P in univr tesfayh weldegebriel Hagos "/home/" MedIAnomaly-DL \
         "NOT a modality" "REPLICATES 4" "four replicate" "not a modality effect"; do
  N=$(grep -rIl "$P" "$OUT" 2>/dev/null | wc -l || true)
  if [ "$N" -eq 0 ]; then printf "  %-22s CLEAN\n" "$P"
  else printf "  %-22s FOUND in %s files\n" "$P" "$N"; FAIL=1
       grep -rl "$P" "$OUT" | sed "s|^|                         |"; fi
done
set -e
echo
echo "=== CONTENTS ==="
echo "  runs:   $(find "$OUT/scores" -name manifest.json | wc -l)"
echo "  scores: $(find "$OUT/scores" -name 'scores.npy' | wc -l)"
echo "  tables: $(ls "$OUT/tables" | wc -l)"
echo "  size:   $(du -sh "$OUT" | cut -f1)"
[ "$FAIL" -eq 0 ] && echo "  STATUS: clean, safe to submit" || { echo "  STATUS: NOT CLEAN"; exit 1; }
