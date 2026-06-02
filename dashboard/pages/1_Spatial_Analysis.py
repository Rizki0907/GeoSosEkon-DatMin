import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg-grid"></div>
        <div class="hero-glow"></div>
        <div class="hero-eyebrow">Layer 3 Analysis</div>
        <h1 class="hero-title">
            Spatial <span class="grad">Autocorrelation</span>
        </h1>
        <p class="hero-sub">
            Spatial autocorrelation metrics analyze geographical clustering of poverty in Indonesia. 
            This identifies hot-spots (High-High) and cold-spots (Low-Low) using spatial weight connectivity and Local Indicators of Spatial Association (LISA).
        </p>
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
                title=dict(font=dict(color='#00D4FF', family='Outfit', size=16)),
                legend=dict(font=dict(color='#94A3B8')),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Configure axes colors
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(255, 255, 255, 0.05)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
                    
            # Colorbar / colorscale customizations
            if hasattr(fig, 'data'):
                for trace in fig.data:
                    if 'colorbar' in trace:
                        trace.colorbar.tickfont = dict(color='#E2E8F0', family='Inter')
                        trace.colorbar.title.font = dict(color='#00D4FF', family='Outfit')
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    fig_pov = load_plotly("plot_poverty_distribution.json")
    if fig_pov:
        st.markdown("<h3 style='font-family: \"Plus Jakarta Sans\"; color: #F8FAFC; font-size:1.15rem; margin-bottom:12px;'>Poverty Distribution Map</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_pov, use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    fig_conn = load_plotly("plot_spatial_connectivity.json")
    if fig_conn:
        st.markdown("<h3 style='font-family: \"Plus Jakarta Sans\"; color: #F8FAFC; font-size:1.15rem; margin-bottom:12px;'>Spatial Connectivity Map (KNN=5)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_conn, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Plus Jakarta Sans\"; color: #F8FAFC; font-size:1.15rem; margin-bottom:12px;'>Global Moran's I Analysis</h3>", unsafe_allow_html=True)

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
    st.markdown("<h3 style='font-family: \"Plus Jakarta Sans\"; color: #F8FAFC; font-size:1.15rem; margin-bottom:12px;'>Local Indicators of Spatial Association (LISA)</h3>", unsafe_allow_html=True)

    fig_lisa_map = load_plotly("plot_lisa_map.json")
    fig_lisa_dist = load_plotly("plot_lisa_distribution.json")
    
    if fig_lisa_map and not fig_lisa_dist:
        st.plotly_chart(fig_lisa_map, use_container_width=True)
    elif fig_lisa_map and fig_lisa_dist:
        col_lisa1, col_lisa2 = st.columns(2)
        with col_lisa1:
            st.plotly_chart(fig_lisa_map, use_container_width=True)
        with col_lisa2:
            st.plotly_chart(fig_lisa_dist, use_container_width=True)

    try:
        df_lisa = pd.read_csv(SPATIAL_OUTPUT / "output_lisa_results.csv")
        st.markdown("<h3 style='font-family: \"Plus Jakarta Sans\"; color: #F8FAFC; margin-top: 1.5rem; margin-bottom: 1rem; font-size:1.15rem;'>LISA Cluster Results</h3>", unsafe_allow_html=True)
        st.dataframe(df_lisa, use_container_width=True)
    except Exception:
        pass


    consistency_path = SPATIAL_OUTPUT / "plot_lisa_consistency.png"
    if consistency_path.exists():
        st.markdown('<div class="card"><div class="card-label">LISA Consistency Distribution</div>', unsafe_allow_html=True)
        st.image(str(consistency_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

