import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton > button {
        background: #5c4be0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 15px;
        font-weight: 500;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #4438c4; color: white; }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        border: 1px solid #e8e8f0;
        margin-top: 1rem;
    }
    .result-price {
        font-size: 2.8rem;
        font-weight: 700;
        color: #5c4be0;
    }
    .result-range {
        font-size: 1.1rem;
        font-weight: 500;
        color: #444;
        margin-top: 0.25rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-budget   { background:#e8f5e9; color:#2e7d32; }
    .badge-mid      { background:#e3f2fd; color:#1565c0; }
    .badge-upper    { background:#fff3e0; color:#e65100; }
    .badge-premium  { background:#fce4ec; color:#880e4f; }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e8e8f0;
        text-align: center;
    }
    .metric-box .metric-val { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
    .metric-box .metric-lbl { font-size: 12px; color: #888; margin-top: 2px; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #999;
        margin-bottom: 0.75rem;
        margin-top: 1.75rem;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label { font-size: 13px !important; font-weight: 500 !important; color: #444 !important; }
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "range_model":    "phone_range_model.joblib",
    "price_model":    "phone_price_model.joblib",
    "target_encoder": "target_encoder.joblib",
    "os_encoder":     "os_encoder.joblib",
    "feature_cols":   "feature_columns.joblib",
}

@st.cache_resource
def load_models():
    missing = [v for v in MODEL_FILES.values() if not os.path.exists(v)]
    if missing:
        return None, missing
    models = {k: joblib.load(v) for k, v in MODEL_FILES.items()}
    return models, []

models, missing_files = load_models()

# ── Header ─────────────────────────────────────────────────────────────────────
col_icon, col_title = st.columns([0.07, 0.93])
with col_icon:
    st.markdown("## 📱")
with col_title:
    st.markdown("## Mobile Price Predictor")
    st.caption("Enter phone specs to estimate the price range and exact price using ML models.")

st.divider()

if missing_files:
    st.error(
        f"**Model files not found:** {', '.join(missing_files)}\n\n"
        "Run `mobile_price_ml.py` first to train and save the models, "
        "then place the `.joblib` files in the same folder as this app."
    )
    st.stop()

# ── Sidebar — input form ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Phone Specifications")
    st.markdown("---")

    brand = st.selectbox("Brand", [
        "Samsung", "Xiaomi", "Oppo", "Vivo", "Realme",
        "OnePlus", "Motorola", "Nokia", "Infinix", "Tecno", "Apple"
    ])

    os_option = st.selectbox("Operating System", ["Android", "iOS", "Other"])

    st.markdown('<p class="section-header">Performance</p>', unsafe_allow_html=True)
    ram_gb            = st.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16])
    storage_gb        = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512])
    processor_speed   = st.slider("Processor Speed (GHz)",1.9, 2.1, 2.3, 2.8, 3.0, 2.9, 2.5, 3.1, 2.4, 2.2, 2.6, 2.7, 2.0, 1.8, 3.2)
    cores             = st.selectbox("CPU Cores", [4, 6, 8])

    st.markdown('<p class="section-header">Camera</p>', unsafe_allow_html=True)
    camera_mp         = st.slider("Main Camera (MP)", 8, 200, 48)
    front_camera_mp   = st.slider("Front Camera (MP)", 5, 50, 16)

    st.markdown('<p class="section-header">Display & Battery</p>', unsafe_allow_html=True)
    screen_size       = st.slider("Screen Size (inch)",6.2, 5.9, 6.5, 6.4, 5.7, 6.3, 6.7, 6.6, 5.8, 5.6, 7.0, 6.0, 6.1, 6.9, 7.1, 6.8, 5.5, 7.2)
    battery_mah       = st.slider("Battery (mAh)",5000, 4500, 4000, 5500, 6000, 3000, 4745)

    st.markdown('<p class="section-header">Connectivity</p>', unsafe_allow_html=True)
    has_5g  = st.radio("5G Support",  ["Yes", "No"], horizontal=True)
    has_nfc = st.radio("NFC Support", ["Yes", "No"], horizontal=True)

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Price")

# ── Build feature row ──────────────────────────────────────────────────────────
def build_input(models, brand, os_option, ram_gb, storage_gb, processor_speed,
                cores, camera_mp, front_camera_mp, screen_size, battery_mah,
                has_5g, has_nfc):

    feature_cols = models["feature_cols"]
    os_encoder   = models["os_encoder"]

    os_encoded = 0
    try:
        os_encoded = int(os_encoder.transform([os_option])[0])
    except Exception:
        pass  # unseen label → default 0

    row = {col: 0 for col in feature_cols}

    # Numeric features — names must match what training produced
    numeric_map = {
        "ram_gb":               ram_gb,
        "storage_gb":           storage_gb,
        "processor_speed_ghz":  processor_speed,
        "cores":                cores,
        "camera_mp":            camera_mp,
        "front_camera_mp":      front_camera_mp,
        "screen_size_inch":     screen_size,
        "battery_mah":          battery_mah,
        "has_5g":               1 if has_5g == "Yes" else 0,
        "has_nfc":              1 if has_nfc == "Yes" else 0,
        "os":                   os_encoded,
    }
    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    # Brand dummy — drop_first=True drops the first brand alphabetically
    brand_col = f"brand_{brand}"
    if brand_col in row:
        row[brand_col] = 1

    return pd.DataFrame([row])[feature_cols]

# ── Badge helper ───────────────────────────────────────────────────────────────
RANGE_META = {
    "budget":                 ("Budget",          "badge-budget",  "💚"),
    "mid-range":              ("Mid-Range",        "badge-mid",     "💙"),
    "price_range_label_mid-range": ("Mid-Range",  "badge-mid",     "💙"),
    "upper mid-range":        ("Upper Mid-Range",  "badge-upper",   "🧡"),
    "price_range_label_upper mid-range": ("Upper Mid-Range","badge-upper","🧡"),
    "premium":                ("Premium",          "badge-premium", "❤️"),
    "price_range_label_premium": ("Premium",       "badge-premium", "❤️"),
}

def get_range_meta(raw_label: str):
    key = raw_label.lower().replace("price_range_label_", "")
    for k, v in RANGE_META.items():
        if key in k or k in key:
            return v
    return (raw_label, "badge-mid", "📱")

# ── Main content ───────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("### Prediction Result")

    if predict_btn:
        with st.spinner("Running models…"):
            X_input = build_input(
                models, brand, os_option, ram_gb, storage_gb,
                processor_speed, cores, camera_mp, front_camera_mp,
                screen_size, battery_mah, has_5g, has_nfc
            )

            # Classification
            range_pred_enc = models["range_model"].predict(X_input)[0]
            range_label    = models["target_encoder"].inverse_transform([range_pred_enc])[0]
            range_proba    = models["range_model"].predict_proba(X_input)[0]

            # Regression
            price_pred = float(models["price_model"].predict(X_input)[0])

        label_text, badge_cls, emoji = get_range_meta(range_label)

        st.markdown(f"""
        <div class="result-card">
            <div style="margin-bottom:0.5rem">
                <span class="badge {badge_cls}">{emoji} {label_text}</span>
            </div>
            <div class="result-price">${price_pred:,.0f}</div>
            <div class="result-range">Estimated retail price</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence breakdown
        st.markdown("#### Confidence per class")
        class_names = models["target_encoder"].classes_
        conf_df = pd.DataFrame({
            "Range":      class_names,
            "Confidence": (range_proba * 100).round(1),
        }).sort_values("Confidence", ascending=False)

        for _, row_data in conf_df.iterrows():
            lbl, bcls, em = get_range_meta(row_data["Range"])
            st.markdown(f"**{em} {lbl}**")
            st.progress(int(row_data["Confidence"]),
                        text=f"{row_data['Confidence']:.1f}%")

    else:
        st.info("👈  Fill in the specs on the left and click **Predict Price**.")

with right:
    st.markdown("### Spec Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{ram_gb}GB</div>
            <div class="metric-lbl">RAM</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{storage_gb}GB</div>
            <div class="metric-lbl">Storage</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{camera_mp}MP</div>
            <div class="metric-lbl">Camera</div></div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{battery_mah}</div>
            <div class="metric-lbl">mAh</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{screen_size}"</div>
            <div class="metric-lbl">Screen</div></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{"5G" if has_5g=="Yes" else "4G"}</div>
            <div class="metric-lbl">Network</div></div>""", unsafe_allow_html=True)

    st.markdown("#### Full spec sheet")
    spec_data = {
        "Spec":  ["Brand", "OS", "RAM", "Storage", "CPU Speed",
                  "Cores", "Main Cam", "Front Cam", "Screen",
                  "Battery", "5G", "NFC"],
        "Value": [brand, os_option, f"{ram_gb} GB", f"{storage_gb} GB",
                  f"{processor_speed} GHz", str(cores),
                  f"{camera_mp} MP", f"{front_camera_mp} MP",
                  f"{screen_size}\"", f"{battery_mah} mAh",
                  has_5g, has_nfc],
    }
    st.dataframe(
        pd.DataFrame(spec_data),
        hide_index=True,
        use_container_width=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Models: XGBoost Classifier (price range) · Random Forest Regressor (price USD)  "
    "— trained on `mobile_price_dataset.csv`"
)
