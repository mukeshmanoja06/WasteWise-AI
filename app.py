# -*- coding: utf-8 -*-
"""
app.py  —  WasteWise AI  |  Main entry point (Home page)
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.ui_utils import apply_theme_css, render_sidebar_branding

st.set_page_config(
    page_title="WasteWise AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS + sidebar branding (applied on every page via shared call) ────
apply_theme_css()
render_sidebar_branding()

# ── Home / Landing page ──────────────────────────────────────────────────────
st.markdown("# ♻️ WasteWise AI")
st.markdown("### Municipal Solid Waste Prediction & Collection Priority Platform")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **📊 Dashboard**
    Overview of waste generation trends, priority distribution, and top cities.
    """)
with col2:
    st.markdown("""
    **🔮 Prediction**
    Estimate municipal solid waste generation for a selected city and year
    using the trained Gradient Boosting model.
    """)
with col3:
    st.markdown("""
    **🏙️ City Analysis**
    Deep-dive into any city's historical waste data and model performance.
    """)

st.markdown("---")
col4, col5 = st.columns(2)
with col4:
    st.markdown("""
    **📈 Analytics**
    Year-over-year trends, city comparisons, and model metrics.
    """)
with col5:
    st.markdown("""
    **🤖 Model**
    Methodology, evaluation results, and known limitations of the ML model.
    """)

st.markdown("---")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; font-size:0.85rem; margin-top:1rem; opacity:0.6;'>
    <strong>WasteWise AI</strong> &nbsp;|&nbsp; Gradient Boosting Regressor &nbsp;|&nbsp;
    45 Indian cities &nbsp;|&nbsp; Observed data: 1999&ndash;2023 &nbsp;|&nbsp;
    RMSE: 581.7&nbsp;t/day &nbsp;|&nbsp; R²: 0.920
</div>
<div style='text-align:center; font-size:0.75rem; margin-top:0.4rem; opacity:0.45;'>
    ⚠️ Predictions are model estimates, not official waste-management data.
    Do not use for operational decisions without independent validation.
</div>
""", unsafe_allow_html=True)
