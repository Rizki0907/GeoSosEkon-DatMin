import streamlit as st
import plotly.io as pio
from pathlib import Path

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">LAYER 4 ANALYSIS</div>
        <div class="hero-title">Causality & Attribution</div>
        <div class="hero-desc">
            Fixed Effects Panel Regression model analyzing socioeconomic causality drivers. 
            By examining longitudinal panel data (2021-2024), we isolate province-specific time-invariant characteristics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PANEL_OUTPUT = BASE_DIR / "LAYER 4 - FIXED EFFECT PANEL REGRESSION" / "panel_output"

    def load_plotly(filename):
        try:
            with open(PANEL_OUTPUT / filename, 'r', encoding='utf-8') as f:
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
            
            # Axis styling
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(255, 255, 255, 0.05)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    fig_trends = load_plotly("plot_panel_trends.json")
    if fig_trends:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Panel Trends (2021-2024)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_trends, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_fe = load_plotly("plot_panel_fe_effects.json")
        if fig_fe:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Individual Fixed Effects by Province</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_fe, use_container_width=True)

    with col2:
        fig_coef = load_plotly("plot_panel_coefs.json")
        if fig_coef:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Coefficient Estimates</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    fig_diag = load_plotly("plot_panel_diagnostics.json")
    if fig_diag:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Residual Diagnostics</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_diag, use_container_width=True)
