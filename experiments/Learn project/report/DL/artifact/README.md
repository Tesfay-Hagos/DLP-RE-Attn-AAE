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
