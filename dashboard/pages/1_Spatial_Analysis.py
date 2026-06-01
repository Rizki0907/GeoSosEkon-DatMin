import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">LAYER 3 ANALYSIS</div>
        <div class="hero-title">Spatial Autocorrelation</div>
        <div class="hero-desc">
            Spatial autocorrelation metrics analyze geographical clustering of poverty in Indonesia. 
            This identifies hot-spots (High-High) and cold-spots (Low-Low) using spatial weight connectivity and Local Indicators of Spatial Association (LISA).
        </div>
    </div>
    """, unsafe_allow_html=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SPATIAL_OUTPUT = BASE_DIR / "LAYER 3 - SPATIAL AUTOKORELASI (MORANS I + LISA MAP)" / "spatial_output"

    def load_plotly(filename):
        try:
            with open(SPATIAL_OUTPUT / filename, 'r', encoding='utf-8') as f:
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
            
            # Configure axes colors
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(255, 255, 255, 0.05)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
                    
            # Colorbar / colorscale customizations if needed (e.g. for choropleths or heatmaps)
            if hasattr(fig, 'data'):
                for trace in fig.data:
                    if 'colorbar' in trace:
                        trace.colorbar.tickfont = dict(color='#E2E8F0', family='Inter')
                        trace.colorbar.title.font = dict(color='#06B6D4', family='Outfit')
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    fig_conn = load_plotly("plot_spatial_connectivity.json")
    if fig_conn:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Spatial Connectivity Map (KNN=5)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_conn, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Global Moran's I Analysis</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_trend = load_plotly("plot_moran_trend.json")
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
    with col2:
        fig_scatter = load_plotly("plot_moran_scatter.json")
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Local Indicators of Spatial Association (LISA)</h3>", unsafe_allow_html=True)

    fig_lisa_map = load_plotly("plot_lisa_map.json")
    if fig_lisa_map:
        st.plotly_chart(fig_lisa_map, use_container_width=True)

    try:
        df_lisa = pd.read_csv(SPATIAL_OUTPUT / "output_lisa_results.csv")
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC; margin-top: 1.5rem; margin-bottom: 1rem;'>LISA Cluster Results</h3>", unsafe_allow_html=True)
        st.dataframe(df_lisa, use_container_width=True)
    except Exception:
        pass
