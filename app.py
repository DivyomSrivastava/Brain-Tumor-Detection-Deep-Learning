"""
------------------------------------------------------------
NeuroVision AI

Single-page Streamlit Dashboard
Brain MRI Tumor Detection + Explainable AI + PDF Report

Author : Divyom Srivastava
------------------------------------------------------------
"""

import os
import tempfile

import streamlit as st

from src.predict import predict_image
from src.gradcam import generate_gradcam
from src.pdf_report import build_pdf_report


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Styling (dark, clean, minimal)
# ============================================================

st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .nv-header {
        text-align: center;
        padding: 28px 0 8px 0;
    }
    .nv-header h1 {
        color: #1f6feb;
        font-size: 2.4rem;
        margin-bottom: 0;
    }
    .nv-header p {
        color: #9aa4b2;
        font-size: 1rem;
        margin-top: 4px;
    }

    .nv-card {
        background-color: #161b22;
        border: 1px solid #262c36;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 26px;
    }

    .nv-pred-label {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }
    .nv-conf-label {
        font-size: 1rem;
        color: #9aa4b2;
    }

    .stButton>button, .stDownloadButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.55em 1.4em;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #1a5fc9;
        color: white;
    }

    hr { border-color: #262c36; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="nv-header">
    <h1>🧠 NeuroVision AI</h1>
    <p>Brain MRI Tumor Detection using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# Upload Section
# ============================================================

st.markdown("### Upload MRI Images")

uploaded_files = st.file_uploader(
    "Drag & drop MRI images (multiple supported)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

analyze_clicked = st.button("🧠 Analyze MRI Scans", use_container_width=False)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# Helper: save uploaded file to a temp path
# ============================================================

def save_temp_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# ============================================================
# Run Analysis + Render Results
# ============================================================

if analyze_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one MRI image first.")
    else:
        for idx, uploaded_file in enumerate(uploaded_files, start=1):

            temp_path = save_temp_file(uploaded_file)

            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                prediction, confidence, probabilities = predict_image(temp_path)
                gradcam_result = generate_gradcam(temp_path, uploaded_file.name)

            st.markdown(f'<div class="nv-card">', unsafe_allow_html=True)

            st.markdown(f"#### MRI {idx} — `{uploaded_file.name}`")

            col_pred, col_conf = st.columns([2, 1])
            with col_pred:
                st.markdown(
                    f'<div class="nv-pred-label">Prediction: {prediction}</div>',
                    unsafe_allow_html=True,
                )
            with col_conf:
                st.markdown(
                    f'<div class="nv-conf-label">Confidence: {confidence:.2f}%</div>',
                    unsafe_allow_html=True,
                )

            st.progress(min(int(confidence), 100))

            st.markdown("")

            # ---- Original vs Grad-CAM ----
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(gradcam_result["original"], caption="Original MRI", use_container_width=True)
            with img_col2:
                st.image(gradcam_result["overlay"], caption="Explainable AI (Grad-CAM)", use_container_width=True)

            st.markdown("")

            # ---- Probability bars ----
            st.markdown("**Class Probabilities**")
            for cls, val in sorted(probabilities.items(), key=lambda x: -x[1]):
                bar_col, val_col = st.columns([5, 1])
                with bar_col:
                    st.progress(min(int(val), 100))
                with val_col:
                    st.write(f"{cls} — {val:.2f}%")

            st.markdown("")

            # ---- PDF Download Button ----
            pdf_bytes = build_pdf_report(
                filename=uploaded_file.name,
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                gradcam_img_path=gradcam_result["overlay"],
                original_img_path=gradcam_result["original"],
            )

            report_name = f"NeuroVision_Report_{os.path.splitext(uploaded_file.name)[0]}.pdf"

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=report_name,
                mime="application/pdf",
                key=f"download_{idx}",
            )

            st.markdown("</div>", unsafe_allow_html=True)

            os.remove(temp_path)


# ============================================================
# Footer
# ============================================================

st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.85rem;'>"
    "NeuroVision AI — for research and educational purposes only. "
    "Not a substitute for professional medical diagnosis."
    "</p>",
    unsafe_allow_html=True,
)