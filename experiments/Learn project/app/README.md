# RE-Attn-AAE Demo App

Streamlit app for showing the ablation results and, specifically, per-image
**localization** quality — the thing a metrics table can't show a professor.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Layout

```
app/
├── app.py              entrypoint — two pages (Overview, Localization Demo)
├── requirements.txt
├── src/
│   ├── data_loader.py   loads everything under data/ (cached)
│   └── visualize.py     figure builders, kept separate from Streamlit calls
├── data/                 input asset bundle (see "Data contract" below)
│   ├── config.json
│   ├── metrics.json
│   ├── loss_history.json
│   ├── scores/           scores_c{1..7}.npy, disc_c{3..7}.npy, binary_test.npy
│   ├── demo/              precomputed per-sample arrays for the localization demo
│   └── weights/           .pth model weights (not used by the app yet — see below)
└── outputs/
    └── images/            PNGs exported from the app land here, named
                            sample{idx:02d}_{condition}_panel.png
```

## Data contract

`data/` is produced by **Cell 19 ("Save streamlit assets")** in
`experiments/re_attn_aae_kaggle-RSNA-ResNet.py`, currently exported as
`streamlit_assets_sample` at the repo root. To refresh this app's data after a
new training run: re-run Cell 19 on Kaggle, download the zip, and replace the
contents of `app/data/` with it (keep the `scores/` subfolder split out from
the top level — this app expects `scores_*.npy`/`disc_*.npy`/`binary_test.npy`
under `data/scores/`, not directly under `data/`).

The `demo/` bundle currently only has precomputed arrays for **C4** and **C6**
(images, reconstructions, SSIM error maps, attention maps, disc scores for 10
sample images) — that's what Cell 19 exports today. Add more conditions to
that cell's export list to compare more of them here.

## Known gap

There's no radiologist ground-truth bounding box in the exported demo bundle
yet, so the app can't currently draw "does the heatmap land inside the box"
overlays — only the raw SSIM/attention heatmaps. If that's wanted for the
professor demo, `test_boxes` (already computed in Cell 2.0 of the main
notebook) needs to be added to Cell 19's export for the same 10 demo sample
indices.

## Live inference (not yet implemented)

`data/weights/` has the trained `.pth` files, which would let the app run
inference on an image you upload live instead of only the 10 precomputed demo
samples — that requires importing the model classes (`CNNEncoder`,
`REAttention`, etc.) and torch, which isn't wired in yet.
