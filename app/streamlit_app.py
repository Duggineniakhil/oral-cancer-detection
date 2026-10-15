"""
Web demo: upload an oral histopathology image (or pick a test image)
and get the predicted class, probabilities and a Grad-CAM heatmap.

Run from the project root (local access only):
    streamlit run app/streamlit_app.py --server.address localhost

Needs a trained model in runs/<run>/best_model.pt (from Kaggle).
RESEARCH PROTOTYPE, not for clinical use.
"""

import sys
from pathlib import Path

# Make the project importable when Streamlit runs this file directly
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from oral_cancer.config import OUTPUT_ROOT, resolve
from oral_cancer.predictor import CLASS_DISPLAY, Predictor


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = {
    "4-class (OSCC · dysplasia · no dysplasia · Normal)": "convnext_tiny_4class_224px",
    "3-class, NDB-UFES (primary benchmark model)": "convnext_tiny_3class_224px",
}

TEST_CSV = "splits/normalized_dataset.csv"
EXAMPLES_PER_CLASS = 4
BAR_COLOR = "#2a78d6"


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner="Loading model...")
def load_predictor(run):
    return Predictor(OUTPUT_ROOT / "runs" / run)


@st.cache_data
def load_examples(classes):
    """A few test-split images per class the model knows (before stain normalization)."""
    try:
        df = pd.read_csv(resolve(TEST_CSV))
    except FileNotFoundError:
        return pd.DataFrame()
    df = df[(df["split"] == "test") & df["class"].isin(classes)]
    return df.groupby("class").sample(n=EXAMPLES_PER_CLASS, random_state=0)[
        ["preprocessed_path", "class", "source"]
    ].reset_index(drop=True)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="Oral histopathology classifier", layout="wide")

st.title("Oral histopathology classifier")
st.caption("ConvNeXt-Tiny · Macenko stain normalization · Grad-CAM")

st.warning(
    "Research prototype for a student project. **Not validated for clinical use.** "
    "Do not use it to make diagnostic decisions.",
    icon="⚠️",
)

available = {label: run for label, run in MODELS.items()
             if (OUTPUT_ROOT / "runs" / run / "best_model.pt").exists()}

if not available:
    st.error("No trained model found in `runs/`. Download `runs.zip` from Kaggle and unzip it into the project folder.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_label = st.radio("Model", list(available))
    run = available[model_label]

    st.header("How it works")
    st.markdown(
        "1. Large images are cut into **512×512 tiles** (never resized); tiles with < 60% tissue are skipped.\n"
        "2. Each tile is **stain-normalized** to the training reference.\n"
        "3. ConvNeXt-Tiny classifies each tile at 224 px; tile probabilities are **averaged**.\n"
        "4. **Grad-CAM** shows the regions that drove each tile's prediction."
    )

    st.header("Known limitations")
    st.markdown(
        "- Trained on NDB-UFES (OSCC, leukoplakia) + a second dataset for **Normal**. "
        "Normal is partly recognised from **dataset-level image properties**, not only tissue. "
        "Heatmaps on Normal often sit at the tile's top edge.\n"
        "- Expects H&E images at **10×/40× objective**, like the training data.\n"
        "- Test performance (3 seeds): 3-class 89.7 ± 1.4% balanced accuracy."
    )

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------
tab_upload, tab_example = st.tabs(["Upload an image", "Try a test image"])

image, true_label = None, None

with tab_upload:
    uploaded = st.file_uploader("H&E image (PNG, JPG or TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

with tab_example:
    examples = load_examples(tuple(load_predictor(run).classes))
    if examples.empty:
        st.info("Test images are not available on this machine (splits/ not found).")
    else:
        options = [f"{CLASS_DISPLAY[r['class']]} — example {i % EXAMPLES_PER_CLASS + 1}"
                   for i, r in examples.iterrows()]
        choice = st.selectbox("Test image (held-out test split)", ["—"] + options)
        if choice != "—":
            row = examples.iloc[options.index(choice)]
            image = Image.open(resolve(row["preprocessed_path"])).convert("RGB")
            true_label = row["class"]
            if uploaded is not None:
                st.caption("Showing the selected test image (clear it to use your upload).")

if image is None:
    st.info("Upload an image or pick a test image to get a prediction.")
    st.stop()

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
predictor = load_predictor(run)

with st.spinner(f"Analysing {image.size[0]}×{image.size[1]} image..."):
    result = predictor.predict(image)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
st.subheader(f"Prediction: {CLASS_DISPLAY[result.prediction]}")

cols = st.columns(2)
cols[0].metric("Confidence", f"{result.confidence:.1%}")
cols[1].metric("Tiles analysed", len(result.tiles))
if true_label is not None:
    if true_label == result.prediction:
        st.success(f"True label (test set): **{CLASS_DISPLAY[true_label]}**: correct")
    else:
        st.error(f"True label (test set): **{CLASS_DISPLAY[true_label]}**: the model was wrong")

for w in result.warnings:
    st.caption(f"ℹ️ {w}")

left, right = st.columns(2)
left.image(image, caption="Input image", width="stretch")
right.image(result.overlay, caption="Grad-CAM on the stain-normalized image (red = most influential)",
            width="stretch")

probs = pd.DataFrame({
    "Class": [CLASS_DISPLAY[c] for c in result.probabilities],
    "Probability": list(result.probabilities.values()),
}).sort_values("Probability", ascending=False)

bars = alt.Chart(probs).mark_bar(color=BAR_COLOR, cornerRadiusEnd=4).encode(
    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1.12]), axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1])),
    y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
    tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
)
labels = bars.mark_text(align="left", dx=4, color="#0b0b0b").encode(text=alt.Text("Probability:Q", format=".1%"))
st.altair_chart((bars + labels).properties(height=180, title="Class probabilities (averaged over tiles)"),
                width="stretch")

if len(result.tiles) > 1:
    with st.expander(f"Per-tile predictions ({len(result.tiles)} tiles)"):
        st.dataframe(pd.DataFrame([
            {"Tile (row, col)": f"({t.row}, {t.col})",
             "Prediction": CLASS_DISPLAY[t.prediction],
             **{CLASS_DISPLAY[c]: f"{p:.1%}" for c, p in t.probabilities.items()}}
            for t in result.tiles
        ]), hide_index=True, width="stretch")

if result.prediction == "Normal":
    st.caption(
        "Note: Normal predictions are affected by a documented dataset shortcut "
        "(see the README). Treat them with extra caution."
    )
