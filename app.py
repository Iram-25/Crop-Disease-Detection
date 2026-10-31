import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = os.environ.get("MODEL_PATH", "tomato_disease_cnn.keras")
LABELS_PATH = os.environ.get("LABELS_PATH", "class_names.json")
IMG_SIZE = 128  # must match img_size used in the notebook
MAX_FILE_MB = 15
LOW_CONFIDENCE = 60.0  # percent

st.set_page_config(
    page_title="Tomato Leaf Doctor",
    page_icon="🍅",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1100px;}
    #MainMenu, footer {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 6px 18px rgba(46,125,50,.25);
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white;}
    .hero p {margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .95;}

    .card {
        background: #ffffff; border-radius: 16px; padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,.08); margin-bottom: 1rem;
        color: #1b1b1b;
    }
    .result-label {font-size: .8rem; letter-spacing: .08em; color: #6b7280; text-transform: uppercase;}
    .result-name {font-size: 1.9rem; font-weight: 700; margin: .2rem 0 .6rem 0;}
    .badge {
        display: inline-block; padding: .25rem .8rem; border-radius: 999px;
        font-weight: 600; font-size: .85rem; color: white;
    }
    .badge.good {background: #2e7d32;}
    .badge.bad {background: #e65100;}

    .bar-row {margin: .65rem 0;}
    .bar-top {display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .2rem;}
    .bar-bg {background: #e5e7eb; border-radius: 999px; height: 10px; overflow: hidden;}
    .bar-fill {height: 10px; border-radius: 999px; background: linear-gradient(90deg, #2e7d32, #81c784);}
    .bar-fill.rest {background: #b0bec5;}

    .info-title {font-weight: 700; margin-bottom: .3rem; color: #1b1b1b;}
    .info-text {color: #374151; line-height: 1.55;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Disease info (matched by keyword in class name) ----------
DISEASE_INFO = {
    "bacterial spot": (
        "Small dark, water-soaked spots on leaves that can turn brown with yellow halos.",
        "Remove infected leaves, avoid overhead watering, rotate crops and use disease-free seed.",
    ),
    "early blight": (
        "Brown spots with concentric rings (target-like) that usually start on older leaves.",
        "Remove affected leaves, mulch the soil, keep plants spaced and rotate crops yearly.",
    ),
    "late blight": (
        "Large, irregular dark patches that spread fast, often with white mold underneath in humid weather.",
        "Act quickly: remove and destroy infected plants and avoid wet foliage. Ask a local expert about fungicides.",
    ),
    "leaf mold": (
        "Pale yellow spots on the upper leaf with olive-green mold underneath, common in humid conditions.",
        "Improve air circulation, reduce humidity and avoid wetting leaves.",
    ),
    "septoria": (
        "Many small circular spots with dark borders and grey centres, mostly on lower leaves.",
        "Remove lower infected leaves, avoid splashing water and rotate crops.",
    ),
    "spider mite": (
        "Tiny yellow speckles and fine webbing, caused by sap-sucking mites in hot, dry weather.",
        "Spray leaves with water, encourage natural predators and consider miticide or neem oil if severe.",
    ),
    "target spot": (
        "Brown spots with light centres and ring patterns on leaves, sometimes on fruit.",
        "Remove infected debris, improve airflow and avoid overhead irrigation.",
    ),
    "yellow leaf curl": (
        "Leaves curl upward, turn yellow and plants become stunted. Spread by whiteflies.",
        "Control whiteflies, remove infected plants and use resistant varieties.",
    ),
    "mosaic": (
        "Mottled light and dark green patches with distorted leaves, caused by a virus.",
        "Remove infected plants, disinfect tools and wash hands. There is no cure once infected.",
    ),
    "healthy": (
        "The leaf looks healthy, with no clear signs of disease detected.",
        "Keep up regular watering, good airflow and routine checks for early signs of problems.",
    ),
}


def get_info(label: str):
    key_text = label.lower().replace("_", " ")
    for key, value in DISEASE_INFO.items():
        if key in key_text:
            return value
    return None


# ---------- Model helpers ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty(name: str) -> str:
    return name.replace("Tomato___", "").replace("Tomato_", "").replace("_", " ").strip()


def preprocess(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype="float32")  # 0-255; model has Rescaling(1/255) inside
    return np.expand_dims(arr, axis=0)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌿 About")
    st.write(
        "A CNN trained on tomato leaf images that predicts the leaf's condition "
        "from a single photo."
    )
    st.markdown("### 📸 Tips for best results")
    st.markdown(
        "- Use one leaf, filling most of the frame\n"
        "- Good natural light, no heavy shadows\n"
        "- Keep the leaf in focus\n"
        "- Plain background works best"
    )
    st.markdown("### ⚠️ Note")
    st.caption(
        "Predictions come from a model and can be wrong. Check with a qualified "
        "agricultural adviser before treating your crop."
    )

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🍅 Tomato Leaf Doctor</h1>
        <p>Upload a leaf photo and get an instant AI diagnosis of common tomato diseases.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Guard checks ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    st.error(
        f"Missing `{MODEL_PATH}` or `{LABELS_PATH}`. "
        "Place both files beside app.py."
    )
    st.stop()

model = load_model()
class_names = load_labels()

if model.output_shape[-1] != len(class_names):
    st.error("Model and class_names.json do not match. Use files from the same training run.")
    st.stop()

# ---------- Main layout ----------
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown("#### 1. Upload a leaf image")
    uploaded = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded is not None:
        if uploaded.size > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB.")
        else:
            try:
                image = Image.open(uploaded)
                image.load()
                st.image(image, caption="Your leaf", use_container_width=True)
            except Exception:
                st.error("Could not read this image. Try another file.")
                image = None

with right:
    st.markdown("#### 2. Diagnosis")

    if image is None:
        st.markdown(
            """
            <div class="card">
                <div class="info-title">Waiting for an image</div>
                <div class="info-text">Upload a tomato leaf photo on the left and the result will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing leaf..."):
            probs = model.predict(preprocess(image), verbose=0)[0]

        top = np.argsort(probs)[::-1][:3]
        best = int(top[0])
        confidence = float(probs[best]) * 100
        label = pretty(class_names[best])
        is_healthy = "healthy" in label.lower()

        badge_class = "good" if is_healthy else "bad"
        badge_text = "Healthy" if is_healthy else "Disease detected"
        icon = "✅" if is_healthy else "🩺"

        st.markdown(
            f"""
            <div class="card">
                <div class="result-label">Predicted condition</div>
                <div class="result-name">{icon} {label}</div>
                <span class="badge {badge_class}">{badge_text}</span>
                <span style="margin-left:.6rem;color:#374151;">Confidence: <b>{confidence:.1f}%</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence < LOW_CONFIDENCE:
            st.warning("Low confidence. Try a sharper, well-lit photo of a single tomato leaf.")

        # Top 3 bars
        bars = ""
        for rank, i in enumerate(top):
            pct = float(probs[int(i)]) * 100
            fill_class = "bar-fill" if rank == 0 else "bar-fill rest"
            bars += (
                f'<div class="bar-row">'
                f'<div class="bar-top"><span>{pretty(class_names[int(i)])}</span><b>{pct:.1f}%</b></div>'
                f'<div class="bar-bg"><div class="{fill_class}" style="width:{pct:.1f}%"></div></div>'
                f"</div>"
            )
        st.markdown(
            f'<div class="card"><div class="info-title">Top 3 predictions</div>{bars}</div>',
            unsafe_allow_html=True,
        )

        info = get_info(class_names[best])
        if info:
            st.markdown(
                f"""
                <div class="card">
                    <div class="info-title">About this condition</div>
                    <div class="info-text">{info[0]}</div>
                    <div class="info-title" style="margin-top:.9rem;">Suggested care</div>
                    <div class="info-text">{info[1]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.caption("Model prediction only. Consult a qualified agricultural adviser before treatment decisions.")