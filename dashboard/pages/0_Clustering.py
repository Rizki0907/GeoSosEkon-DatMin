import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">LAYER 1 ANALYSIS</div>
        <div class="hero-title">Provincial Typology</div>
        <div class="hero-desc">
            Socio-economic clustering of Indonesian provinces using Gaussian Mixture Models (GMM) 
            projected onto 2D and 3D UMAP/PCA spaces. This categorizes regional profiles to optimize policy interventions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CLUSTER_OUTPUT = BASE_DIR / "LAYER 1 - CLUSTERING" / "cluster_output"

    def load_plotly(filename):
        try:
            with open(CLUSTER_OUTPUT / filename, 'r', encoding='utf-8') as f:
                json_str = f.read().replace('"heatmapgl"', '"heatmap"')
            fig = pio.from_json(json_str)
            
            # Apply premium dark re-theming
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', family='Inter'),
                title=dict(font=dict(color='#06B6D4', family='Outfit', size=16)),
                legend=dict(font=dict(color='#94A3B8')),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Adjust gridlines and backgrounds for 2D/3D axes
            if 'scene' in fig.layout:
                scene = fig.layout.scene
                for axis in [scene.xaxis, scene.yaxis, scene.zaxis]:
                    axis.gridcolor = 'rgba(255, 255, 255, 0.05)'
                    axis.linecolor = 'rgba(255, 255, 255, 0.1)'
                    axis.backgroundcolor = 'rgba(0, 0, 0, 0)'
                    axis.zerolinecolor = 'rgba(255, 255, 255, 0.1)'
            else:
                for ax_name in ['xaxis', 'yaxis']:
                    if ax_name in fig.layout:
                        fig.layout[ax_name].gridcolor = 'rgba(255, 255, 255, 0.05)'
                        fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                        fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    col1, col2 = st.columns(2)

    with col1:
        fig_2d = load_plotly("plot_gmm_comparison_2d.json")
        if fig_2d:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>PCA & UMAP 2D Projection</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_2d, use_container_width=True)

    with col2:
        fig_3d = load_plotly("plot_gmm_comparison_3d.json")
        if fig_3d:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>PCA & UMAP 3D Projection</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    fig_pca = load_plotly("plot_pca_scree.json")
    if fig_pca:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>PCA Variance Explained (Scree Plot)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    try:
        df_clusters = pd.read_csv(CLUSTER_OUTPUT / "output_province_clusters.csv")
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC; margin-bottom: 1rem;'>GMM Cluster Assignments</h3>", unsafe_allow_html=True)
        st.dataframe(df_clusters, use_container_width=True)
    except Exception:
        pass
