# -*- coding: utf-8 -*-
"""
src/ui_utils.py
---------------
Shared UI utility functions for all Streamlit pages.
Includes theme toggle and branded sidebar.
"""
from __future__ import annotations
import streamlit as st
from src.priority import THRESHOLDS, PRIORITY_META

# ── Latest observed year (the model's training horizon) ─────────────────────
LAST_OBSERVED_YEAR = 2022   # fiscal year 2022-23 is the last in the dataset


def apply_theme_css() -> None:
    """
    Inject CSS variables that adapt to the current theme stored in
    st.session_state["theme"].  Call once per page, before any content.

    Light theme: warm off-white background, dark text, high contrast charts.
    Dark theme:  deep slate background, light text, adjusted card surfaces.
    """
    dark = st.session_state.get("theme", "light") == "dark"

    if dark:
        bg        = "#0e1117"
        sbg       = "#1a1f2e"
        text      = "#e8ecf0"
        muted     = "#8d96a3"
        border    = "rgba(255,255,255,0.10)"
        hr_color  = "rgba(255,255,255,0.10)"
        card_bg   = "#1a1f2e"
    else:
        bg        = "#f8fafb"
        sbg       = "#ffffff"
        text      = "#1a1a2e"
        muted     = "#555f6d"
        border    = "rgba(0,0,0,0.07)"
        hr_color  = "rgba(0,0,0,0.08)"
        card_bg   = "#ffffff"

    st.markdown(f"""
<style>
/* ── Root overrides ──────────────────────────────────────────────────────── */
.stApp                          {{ background-color: {bg}; color: {text}; }}
section[data-testid="stSidebar"]{{ background-color: {sbg}; }}
.stApp p, .stApp li             {{ color: {text}; }}
.stApp h1,.stApp h2,.stApp h3,
.stApp h4,.stApp h5,.stApp h6   {{ color: {text}; }}
.stApp .stCaption               {{ color: {muted}; }}

/* ── Metric cards ────────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {{
    background: {card_bg};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid {border};
}}
div[data-testid="stMetricLabel"] > div {{ color: {muted} !important; font-size:0.82rem; }}
div[data-testid="stMetricValue"]        {{ color: {text} !important; }}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {{ border: none; border-top: 1px solid {hr_color}; margin: 1rem 0; }}

/* ── DataFrames / Tables ─────────────────────────────────────────────────── */
.stDataFrame {{ border-radius: 8px; }}

/* ── Main content padding ────────────────────────────────────────────────── */
.main .block-container {{ padding-top: 1.5rem; }}

/* ── Sidebar nav ─────────────────────────────────────────────────────────── */
[data-testid="stSidebarNav"] {{ padding-top: 0.5rem; }}
</style>
""", unsafe_allow_html=True)


def render_sidebar_branding() -> None:
    """
    Render the WasteWise AI brand block and Light/Dark theme toggle in the
    sidebar.  Call once per page (before page-specific content).
    """
    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 0.6rem 0 0.2rem 0;">
                <div style="font-size:1.25rem; font-weight:700; letter-spacing:-0.3px;">
                    ♻️ WasteWise AI
                </div>
                <div style="font-size:0.78rem; opacity:0.65; margin-top:1px;">
                    Municipal Waste Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Theme toggle ────────────────────────────────────────────────────
        if "theme" not in st.session_state:
            st.session_state["theme"] = "light"

        current = st.session_state["theme"]
        toggle_label = "🌙 Switch to Dark mode" if current == "light" else "☀️ Switch to Light mode"
        if st.button(toggle_label, use_container_width=True, key="theme_toggle"):
            st.session_state["theme"] = "dark" if current == "light" else "light"
            st.rerun()

        st.markdown("---")


def priority_badge(label: str) -> str:
    """Return an HTML badge string for a priority label."""
    meta = PRIORITY_META.get(label, {"color": "#888", "emoji": "⚪", "description": label})
    return (
        f'<span style="background:{meta["color"]};color:#fff;'
        f'padding:4px 12px;border-radius:12px;font-weight:600;'
        f'font-size:0.85rem;letter-spacing:0.3px">{meta["emoji"]} {label}</span>'
    )


def metric_card(label: str, value: str, delta: str = "", color: str = "#1a1a2e") -> None:
    """Render a styled metric card using st.metric."""
    st.metric(label=label, value=value, delta=delta if delta else None)


def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def info_box(text: str, kind: str = "info") -> None:
    """Render a styled info / warning / error box."""
    if kind == "warning":
        st.warning(text)
    elif kind == "error":
        st.error(text)
    elif kind == "success":
        st.success(text)
    else:
        st.info(text)


def threshold_explainer() -> None:
    """Show a standard threshold explanation block."""
    st.caption(
        f"**Priority thresholds** (frozen from training data, n={THRESHOLDS.n_train}):  "
        f"🟢 LOW ≤ {THRESHOLDS.p33:,.2f} t/day  |  "
        f"🟡 MEDIUM ≤ {THRESHOLDS.p67:,.1f} t/day  |  "
        f"🔴 HIGH > {THRESHOLDS.p67:,.1f} t/day"
    )


def extrapolation_warning(year: int) -> None:
    """
    Display a visible warning when the requested year is beyond the last
    observed training period (year_start > LAST_OBSERVED_YEAR).
    """
    if year > LAST_OBSERVED_YEAR:
        st.warning(
            f"**Extrapolation:** Year {year} is beyond the latest observed training period "
            f"(last observed: {LAST_OBSERVED_YEAR}, fiscal year 2022-23). "
            "This result is a model estimate based on extrapolated time trends and "
            "**should be validated before any operational use**."
        )
