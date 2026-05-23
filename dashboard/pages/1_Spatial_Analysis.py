import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Spatial Autocorrelation", layout="wide")

st.markdown("<h2 style='text-align: center;'>🗺️ Spatial Autocorrelation (Moran's I & LISA)</h2>", unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPATIAL_OUTPUT = BASE_DIR / "LAYER 3 - SPATIAL AUTOKORELASI (MORANS I + LISA MAP)" / "spatial_output"

def load_plotly(filename):
    try:
        return pio.read_json(SPATIAL_OUTPUT / filename)
    except Exception as e:
        return None

fig_conn = load_plotly("plot_spatial_connectivity.json")
if fig_conn:
    st.markdown("### 🌐 Spatial Connectivity Map (KNN=5)")
    st.plotly_chart(fig_conn, use_container_width=True)

st.markdown("---")
st.markdown("### 🔍 Global Moran's I")

col1, col2 = st.columns(2)
with col1:
    fig_trend = load_plotly("plot_moran_trend.json")
    if fig_trend:
        st.plotly_chart(fig_trend, use_container_width=True)
with col2:
    fig_scatter = load_plotly("plot_moran_scatter.json")
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.markdown("### 📍 Local Indicators of Spatial Association (LISA)")

fig_lisa_map = load_plotly("plot_lisa_map.json")
if fig_lisa_map:
    st.plotly_chart(fig_lisa_map, use_container_width=True)

try:
    df_lisa = pd.read_csv(SPATIAL_OUTPUT / "output_lisa_results.csv")
    st.markdown("### 📋 LISA Cluster Results")
    st.dataframe(df_lisa, use_container_width=True)
except Exception:
    pass
