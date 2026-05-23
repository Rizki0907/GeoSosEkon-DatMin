import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Provincial Typology", layout="wide")

st.markdown("<h2 style='text-align: center;'>🎯 Provincial Typology (Clustering)</h2>", unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLUSTER_OUTPUT = BASE_DIR / "LAYER 1 - CLUSTERING" / "cluster_output"

def load_plotly(filename):
    try:
        with open(CLUSTER_OUTPUT / filename, 'r', encoding='utf-8') as f:
            json_str = f.read().replace('"heatmapgl"', '"heatmap"')
        return pio.from_json(json_str)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

col1, col2 = st.columns(2)

with col1:
    fig_2d = load_plotly("plot_gmm_comparison_2d.json")
    if fig_2d:
        st.markdown("### PCA & UMAP (2D Projection)")
        st.plotly_chart(fig_2d, use_container_width=True)

with col2:
    fig_3d = load_plotly("plot_gmm_comparison_3d.json")
    if fig_3d:
        st.markdown("### PCA & UMAP (3D Projection)")
        st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

fig_pca = load_plotly("plot_pca_scree.json")
if fig_pca:
    st.markdown("### PCA Variance Explained")
    st.plotly_chart(fig_pca, use_container_width=True)

try:
    df_clusters = pd.read_csv(CLUSTER_OUTPUT / "output_province_clusters.csv")
    st.markdown("### 📋 Cluster Assignments")
    st.dataframe(df_clusters, use_container_width=True)
except Exception:
    pass
