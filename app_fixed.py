import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Wound Healing Predictor",
    page_icon="🩹",
    layout="centered"  # mobile ke liye centered better hai
)

# ── Custom CSS — Responsive ───────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Main container */
.main .block-container {
    padding: 1.5rem 1rem;
    max-width: 900px;
}

/* Title */
h1 {
    font-size: clamp(1.4rem, 4vw, 2.2rem) !important;
    text-align: center;
    color: #2c3e50;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: clamp(0.8rem, 2.5vw, 1rem);
    margin-bottom: 1.5rem;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 0.8rem;
    border: 1px solid #e0e0e0;
    text-align: center;
}

[data-testid="metric-container"] label {
    font-size: clamp(0.7rem, 2vw, 0.9rem) !important;
    color: #555 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: clamp(1rem, 3vw, 1.5rem) !important;
    font-weight: 700 !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #3498db;
    border-radius: 12px;
    padding: 1rem;
    background: #f0f8ff;
}

/* Image captions */
.stImage > div > div > p {
    font-size: clamp(0.7rem, 2vw, 0.85rem);
    text-align: center;
    color: #666;
}

/* Alert boxes */
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 10px;
    font-size: clamp(0.8rem, 2vw, 1rem);
}

/* Severity badge */
.badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-weight: bold;
    font-size: clamp(0.9rem, 2.5vw, 1.1rem);
    margin: 0.5rem 0;
}
.badge-mild     { background: #d5f5e3; color: #1e8449; }
.badge-moderate { background: #fef9e7; color: #b7950b; }
.badge-severe   { background: #fde8e8; color: #c0392b; }
.badge-critical { background: #2c0000; color: #ff4444; }

/* Responsive columns — stack on mobile */
@media (max-width: 640px) {
    [data-testid="column"] {
        min-width: 100% !important;
        margin-bottom: 0.8rem;
    }
}

/* Progress bar */
.stProgress > div > div {
    border-radius: 10px;
    height: 12px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

# ── Custom Loss Functions ─────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1 - dice_coefficient(y_true, y_pred)
    return bce + dice

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Users\subhangini\OneDrive\Desktop\Wound-Segmentation\unet_best.keras",
        custom_objects={
            'bce_dice_loss':    bce_dice_loss,
            'dice_coefficient': dice_coefficient,
            'iou_metric':       iou_metric
        }
    )

# ── Core Functions ────────────────────────────────────────
def predict_mask(model, img_array):
    img  = cv2.resize(img_array, (256, 256))
    inp  = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
    pred = model.predict(inp, verbose=0)[0, :, :, 0]
    return (pred > 0.5).astype(np.uint8)

def get_wound_area_pct(mask):
    total = mask.shape[0] * mask.shape[1]
    wound = cv2.countNonZero(mask)
    return round((wound / total) * 100, 2)

def estimate_healing(wound_area_pct):
    if wound_area_pct <= 0:
        return {"severity": "No Wound", "days": 0,
                "range": "N/A", "advice": "No wound detected in image.",
                "badge": ""}
    elif wound_area_pct <= 5:
        days = int(7  + wound_area_pct * 1.4)
        return {"severity": "Mild", "days": days,
                "range": f"{days-2}–{days+3} days",
                "advice": "Minor wound. Clean daily and keep covered. No doctor needed unless infected.",
                "badge": "badge-mild"}
    elif wound_area_pct <= 15:
        days = int(14 + wound_area_pct * 1.6)
        return {"severity": "Moderate", "days": days,
                "range": f"{days-3}–{days+5} days",
                "advice": "Change dressing every 2 days. See a doctor if no improvement in 1 week.",
                "badge": "badge-moderate"}
    elif wound_area_pct <= 30:
        days = int(30 + wound_area_pct * 1.8)
        return {"severity": "Severe", "days": days,
                "range": f"{days-5}–{days+10} days",
                "advice": "Daily dressing change needed. Consult a doctor immediately.",
                "badge": "badge-severe"}
    else:
        days = int(60 + wound_area_pct * 2.0)
        return {"severity": "Critical", "days": days,
                "range": f"{days-7}–{days+14} days",
                "advice": "Emergency medical attention required. Do not delay.",
                "badge": "badge-critical"}

def make_overlay(img_rgb, mask):
    resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
    overlay = img_rgb.copy()
    overlay[resized == 1] = (
        overlay[resized == 1] * 0.5 +
        np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)
    return overlay

# ── UI ────────────────────────────────────────────────────
st.markdown("<h1>🩹 Wound Healing Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a wound image to get mask, severity & healing estimate</p>',
    unsafe_allow_html=True
)

model = load_model()
# ── Patient Details ───────────────────────────────────────
st.markdown("### 👤 Patient Details")

col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name = st.text_input("Patient Name")
with col_b:
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=20)
with col_c:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.divider()

uploaded = st.file_uploader(
    "📤 Upload Wound Image", type=["jpg", "jpeg", "png"]
)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🔍 Analyzing wound..."):
        mask     = predict_mask(model, img_rgb)
        area_pct = get_wound_area_pct(mask)
        result   = estimate_healing(area_pct)
        overlay  = make_overlay(img_rgb, mask)
        # ── Patient Info Card ─────────────────────────
        if patient_name:
            st.markdown(f"""
            <div style='background:#f0f8ff; border-radius:12px;
                        padding:1rem; border-left:4px solid #3498db;
                        margin-bottom:1rem;'>
                <b>👤 Patient:</b> {patient_name} &nbsp;|&nbsp;
                <b>🎂 Age:</b> {patient_age} yrs &nbsp;|&nbsp;
                <b>⚧ Gender:</b> {patient_gender}
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        

    # ── Images — responsive 3 col ──
    st.markdown("### 🖼️ Image Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb,      caption="Original",     use_container_width=True)
    with col2:
        st.image(mask * 255,   caption="Wound Mask",   use_container_width=True, clamp=True)
    with col3:
        st.image(overlay,      caption="Overlay",      use_container_width=True)

    st.divider()

    # ── Metrics — responsive 4 col ──
    st.markdown("### 📊 Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🩸 Wound Area",      f"{area_pct}%")
    m2.metric("⚠️ Severity",        result["severity"])
    m3.metric("📅 Est. Healing",    f"~{result['days']} days")
    m4.metric("📆 Range",           result["range"])

    st.divider()

    # ── Severity Badge + Advice ──
    st.markdown("### 🏥 Medical Advice")
    st.markdown(
        f'<span class="badge {result["badge"]}">{result["severity"]}</span>',
        unsafe_allow_html=True
    )

    severity = result["severity"]
    if severity == "No Wound":
        st.info(f"ℹ️ {result['advice']}")
    elif severity == "Mild":
        st.success(f"✅ {result['advice']}")
    elif severity == "Moderate":
        st.warning(f"⚠️ {result['advice']}")
    else:
        st.error(f"🚨 {result['advice']}")

    # ── Healing Progress ──
    st.divider()
    st.markdown("### 📈 Healing Timeline")
    st.caption(f"Expected recovery in **{result['range']}** (~{result['days']} days)")
    progress = min(1.0, area_pct / 35)
    st.progress(progress)

    # ── Wound Stats expander ──
    with st.expander("🔬 View Detailed Stats"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Wound Area | {area_pct}% of image |
        | Severity Level | {result['severity']} |
        | Estimated Days | {result['days']} days |
        | Healing Range | {result['range']} |
        | Advice | {result['advice']} |
        """)

    # ── Footer ──
    st.markdown(
        '<div class="footer">Wound Healing Predictor • B.Tech Final Year Project • Subhangini Patra</div>',
        unsafe_allow_html=True
    )

else:
    st.info("👆 Please upload a wound image to begin analysis.")
    st.markdown("""
    <div style='text-align:center; color:#aaa; margin-top:3rem;'>
        <div style='font-size:4rem;'>🩹</div>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)