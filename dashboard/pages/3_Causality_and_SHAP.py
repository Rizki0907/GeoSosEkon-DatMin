import streamlit as st
import plotly.io as pio
from pathlib import Path

st.set_page_config(page_title="Causality & Panel Data", layout="wide")

st.markdown("<h2 style='text-align: center;'>🔗 Causality & Attribution (Fixed Effects Panel Regression)</h2>", unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PANEL_OUTPUT = BASE_DIR / "LAYER 4 - FIXED EFFECT PANEL REGRESSION" / "panel_output"

def load_plotly(filename):
    try:
        return pio.read_json(PANEL_OUTPUT / filename)
    except Exception as e:
        return None

fig_trends = load_plotly("plot_panel_trends.json")
if fig_trends:
    st.markdown("### 📊 Panel Trends (2021-2024)")
    st.plotly_chart(fig_trends, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    fig_fe = load_plotly("plot_panel_fe_effects.json")
    if fig_fe:
        st.markdown("### 🏛️ Individual Fixed Effects by Province")
        st.plotly_chart(fig_fe, use_container_width=True)

with col2:
    fig_coef = load_plotly("plot_panel_coefs.json")
    if fig_coef:
        st.markdown("### ⚖️ Coefficient Estimates")
        st.plotly_chart(fig_coef, use_container_width=True)

st.markdown("---")
fig_diag = load_plotly("plot_panel_diagnostics.json")
if fig_diag:
    st.markdown("### 🛠️ Residual Diagnostics")
    st.plotly_chart(fig_diag, use_container_width=True)
