"""RE-Attn-AAE demo app — for showing the ablation results and, most
importantly, per-image localization quality (where the model thinks the
anomaly is vs. where a plain baseline thinks it is).

Run with:  streamlit run app.py   (from this app/ directory)

Data contract: everything under ./data/ — see src/data_loader.py for the
exact files expected. That folder is produced by Cell 19 ("Save streamlit
assets") in experiments/re_attn_aae_kaggle-RSNA-ResNet.py and copied here.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import visualize as viz

OUTPUT_IMAGES_DIR = Path(__file__).resolve().parent / "outputs" / "images"

st.set_page_config(page_title="RE-Attn-AAE Demo", layout="wide")


def page_overview():
    st.header("Ablation Overview")
    config = dl.load_config()
    metrics = dl.load_metrics()
    palette = config.get("palette", {})

    conds = dl.available_conditions()
    if not conds:
        st.info(
            "No results exported yet. Run the export cell in the notebook "
            "(after at least one condition is `is_done()`) and unzip its "
            "output into `app/data/` — see README.md."
        )
        return

    rows = {c: metrics[c] for c in conds}
    df = pd.DataFrame(rows).T[["auc_roc", "auc_pr", "f1"]]
    df.index.name = "condition"
    st.caption(f"Showing {len(conds)} finished condition(s): {', '.join(conds)}")

    st.subheader("Metrics table")
    st.dataframe(df.style.format("{:.3f}"))

    st.subheader("Metric comparison")
    metric = st.selectbox("Metric", ["auc_roc", "auc_pr", "f1"], index=0)
    st.pyplot(viz.metrics_bar_chart(df, metric, palette))

    st.subheader("Training loss curves")
    loss_history = dl.load_loss_history()
    st.pyplot(viz.loss_curve_chart(loss_history, palette))


def page_localization_demo():
    st.header("Localization Demo — where does the model think the anomaly is?")
    st.caption(
        "Each row compares one condition's reconstruction-error heatmap for the "
        "same input image. A stronger model concentrates error on the actual "
        "pneumonia region instead of spreading it across normal anatomy (ribs, "
        "heart border)."
    )

    bundle = dl.load_demo_bundle()
    if bundle is None:
        st.info(
            "No localization demo bundle exported yet. This needs per-sample "
            "reconstruction/SSIM-map arrays (data/demo/images.npy etc.), which "
            "the current export cell doesn't produce for scalar-score-only "
            "conditions like a plain C1 run — see README.md 'Known gap'."
        )
        return

    images, labels = bundle["images"], bundle["labels"]
    n = images.shape[0]

    idx = st.slider("Demo sample index", 0, n - 1, 0)
    label_txt = "Lung Opacity (anomaly)" if labels[idx] == 1 else "Normal"
    st.markdown(f"**Sample {idx}** — ground-truth class: **{label_txt}**")

    conds_available = sorted(bundle["conditions"].keys())
    if not conds_available:
        st.warning("No per-condition demo arrays found in data/demo/.")
        return

    chosen = st.multiselect("Conditions to compare", conds_available, default=conds_available)

    for cond in chosen:
        c = bundle["conditions"][cond]
        recon = c["recon"][idx]
        ssim_map = c["ssim"][idx]
        attn_map = c["attn"][idx] if c["attn"] is not None else None
        disc_score = float(c["disc"][idx]) if c["disc"] is not None else None
        scores = dl.load_scores(cond)
        score = float(scores[idx]) if scores is not None and idx < len(scores) else None

        st.markdown(f"#### {cond}")
        fig = viz.sample_panel(
            images[idx], recon, ssim_map, attn_map, cond, label_txt, score, disc_score
        )
        st.pyplot(fig)

        if st.button(f"Save this panel ({cond}, sample {idx})", key=f"save_{cond}_{idx}"):
            out_name = f"sample{idx:02d}_{cond.lower()}_panel"
            path = viz.save_figure(fig, OUTPUT_IMAGES_DIR, out_name)
            st.success(f"Saved to {path.relative_to(Path(__file__).resolve().parent)}")


def main():
    st.sidebar.title("RE-Attn-AAE")
    page = st.sidebar.radio("Page", ["Ablation Overview", "Localization Demo"])
    if page == "Ablation Overview":
        page_overview()
    else:
        page_localization_demo()


if __name__ == "__main__":
    main()
