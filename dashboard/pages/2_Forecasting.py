import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Poverty Forecasting", layout="wide")

st.markdown("<h2 style='text-align: center;'>📈 Projections & Predictions (XGBoost & SHAP)</h2>", unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FORECAST_OUTPUT = BASE_DIR / "LAYER 2 - FORECAST + SHAP" / "forecast_output"

def load_plotly(filename):
    try:
        with open(FORECAST_OUTPUT / filename, 'r', encoding='utf-8') as f:
            json_str = f.read().replace('"heatmapgl"', '"heatmap"')
        return pio.from_json(json_str)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

fig_spag = load_plotly("plot_forecast_spaghetti.json")
if fig_spag:
    st.markdown("### 🔮 Provincial Poverty Projection (2025-2026)")
    st.plotly_chart(fig_spag, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig_avg = load_plotly("plot_forecast_average_trend.json")
    if fig_avg:
        st.markdown("### National Average Trend")
        st.plotly_chart(fig_avg, use_container_width=True)
        
with col2:
    fig_comb = load_plotly("plot_forecast_combined_trend.json")
    if fig_comb:
        st.markdown("### Combined Trend Overview")
        st.plotly_chart(fig_comb, use_container_width=True)

st.markdown("---")
st.markdown("### 🧠 SHAP Interpretability")

fig_shap_0 = load_plotly("plot_shap_forecast_cluster_0.json")
fig_shap_1 = load_plotly("plot_shap_forecast_cluster_1.json")

if fig_shap_0 or fig_shap_1:
    scol1, scol2 = st.columns(2)
    with scol1:
        if fig_shap_0: st.plotly_chart(fig_shap_0, use_container_width=True)
    with scol2:
        if fig_shap_1: st.plotly_chart(fig_shap_1, use_container_width=True)
