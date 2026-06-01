import streamlit as st
import plotly.io as pio
from pathlib import Path

def show():
    # Page Hero matching the premium design guidelines
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg-grid"></div>
        <div class="hero-glow"></div>
        <div class="hero-eyebrow">Layer 4 Analysis</div>
        <h1 class="hero-title">
            Causality & <br><span class="grad">Attribution</span>
        </h1>
        <p class="hero-sub">
            Longitudinal panel data modeling (2021-2024) using Fixed Effects Panel Regression to analyze 
            socioeconomic drivers and isolate province-specific time-invariant characteristics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PANEL_OUTPUT = BASE_DIR / "LAYER 4 - FIXED EFFECT PANEL REGRESSION" / "panel_output"

    def load_plotly(filename):
        try:
            with open(PANEL_OUTPUT / filename, 'r', encoding='utf-8') as f:
                json_str = f.read().replace('"heatmapgl"', '"heatmap"')
            fig = pio.from_json(json_str)
            
            # Apply premium dark re-theming consistent with style.css
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', family='Space Grotesk, sans-serif', size=11),
                title=dict(font=dict(color='#00D4FF', family='Plus Jakarta Sans', size=16)),
                legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='#94A3B8')),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Axis styling
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(30, 45, 74, 0.6)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].tickfont = dict(size=10, color='#94A3B8')
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    # Section 1: Panel Trends
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Longitudinal Socioeconomic Trends</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    fig_trends = load_plotly("plot_panel_trends.json")
    if fig_trends:
        st.markdown('<div class="card"><div class="card-label">Panel Trends (2021-2024)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_trends, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Section 1.5: Data Exploration & Multicollinearity
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Variable Correlation & Multicollinearity</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col0_1, col0_2 = st.columns(2)
    with col0_1:
        fig_corr = load_plotly("plot_panel_correlation.json")
        if fig_corr:
            st.markdown('<div class="card"><div class="card-label">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with col0_2:
        fig_vif = load_plotly("plot_panel_vif.json")
        if fig_vif:
            st.markdown('<div class="card"><div class="card-label">Variance Inflation Factor (VIF)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_vif, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: Model Estimation & Effects
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Fixed Effects Estimation Results</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_fe = load_plotly("plot_panel_fe_effects.json")
        if fig_fe:
            st.markdown('<div class="card"><div class="card-label">Individual Fixed Effects by Province</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_fe, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        fig_coef = load_plotly("plot_panel_coefs.json")
        if fig_coef:
            st.markdown('<div class="card"><div class="card-label">Model Coefficient Estimates</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_coef, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Section 3: Diagnostic Validation
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Residual Diagnostics</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    fig_diag = load_plotly("plot_panel_diagnostics.json")
    if fig_diag:
        st.markdown('<div class="card"><div class="card-label">Residual Diagnostics & Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_diag, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
