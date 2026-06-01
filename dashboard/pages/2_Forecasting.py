import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path
import joblib
import numpy as np

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">LAYER 2 ANALYSIS</div>
        <div class="hero-title">Projections & Predictions (XGBoost & SHAP)</div>
        <div class="hero-desc">
            Poverty rate forecasting for the years 2025-2026. This layer features spaghetti trend projections, 
            SHAP model interpretability plots, and an interactive machine learning predictor to simulate policy impact.
        </div>
    </div>
    """, unsafe_allow_html=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    FORECAST_OUTPUT = BASE_DIR / "LAYER 2 - FORECAST + SHAP" / "forecast_output"

    def load_plotly(filename):
        try:
            with open(FORECAST_OUTPUT / filename, 'r', encoding='utf-8') as f:
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
            
            # Axes customization
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(255, 255, 255, 0.05)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    fig_spag = load_plotly("plot_forecast_spaghetti.json")
    if fig_spag:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Provincial Poverty Projection (2025-2026)</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_spag, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_avg = load_plotly("plot_forecast_average_trend.json")
        if fig_avg:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>National Average Trend</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_avg, use_container_width=True)
            
    with col2:
        fig_comb = load_plotly("plot_forecast_combined_trend.json")
        if fig_comb:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Combined Trend Overview</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_comb, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>SHAP Model Interpretability (by Cluster)</h3>", unsafe_allow_html=True)

    fig_shap_0 = load_plotly("plot_shap_forecast_cluster_0.json")
    fig_shap_1 = load_plotly("plot_shap_forecast_cluster_1.json")

    if fig_shap_0 or fig_shap_1:
        scol1, scol2 = st.columns(2)
        with scol1:
            if fig_shap_0: 
                st.plotly_chart(fig_shap_0, use_container_width=True)
        with scol2:
            if fig_shap_1: 
                st.plotly_chart(fig_shap_1, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Interactive Poverty Predictor</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Simulate the impact of changing socio-economic indicators on the projected poverty rate using our trained Ensemble Model (XGBoost, LightGBM, Random Forest).</p>", unsafe_allow_html=True)

    model_path = FORECAST_OUTPUT / "forecasting_models.joblib"
    if model_path.exists():
        models = joblib.load(model_path)
        
        # Enclose form inside a container
        with st.container():
            with st.form("forecast_form"):
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    poverty_rate = st.number_input("Poverty Rate (Previous Year) %", value=10.0, step=0.1)
                    hdi = st.number_input("HDI (Human Development Index)", value=70.0, step=0.1)
                    log_poverty_line = st.number_input("Poverty Line (Log scale)", value=13.0, step=0.1)
                with fcol2:
                    tpt = st.number_input("TPT (Open Unemployment) %", value=5.0, step=0.1)
                    tpak = st.number_input("TPAK (Labor Force Part.) %", value=65.0, step=0.1)
                with fcol3:
                    aps_1315 = st.number_input("School Part. Rate (13-15 yrs) %", value=95.0, step=0.1)
                    mys = st.number_input("Mean Years of Schooling", value=8.5, step=0.1)
                    eys = st.number_input("Expected Years of Schooling", value=13.0, step=0.1)
                    
                submitted = st.form_submit_button("Predict Poverty Rate", use_container_width=True)
                
                if submitted:
                    X_input = np.array([[poverty_rate, hdi, tpt, tpak, aps_1315, mys, eys, log_poverty_line]])
                    
                    rf_pred = models['rf_model'].predict(X_input)[0]
                    xgb_pred = models['xgb_model'].predict(X_input)[0]
                    lgb_pred = models['lgb_model'].predict(X_input)[0]
                    ew = models['ensemble_weights']
                    
                    final_pred = (rf_pred * ew['rf']) + (xgb_pred * ew['xgb']) + (lgb_pred * ew['lgb'])
                    
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 10px; margin-top: 15px; text-align: center;">
                        <h4 style="margin: 0 0 5px 0; color: #10B981; font-family: 'Outfit';">Simulation Result</h4>
                        <p style="margin: 0; font-size: 1.15rem; color: #E2E8F0;">
                            Predicted Poverty Rate: <strong>{final_pred:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Show Model Breakdown"):
                        st.write(f"Random Forest Prediction: {rf_pred:.2f}% (Weight: {ew['rf']:.2f})")
                        st.write(f"XGBoost Prediction: {xgb_pred:.2f}% (Weight: {ew['xgb']:.2f})")
                        st.write(f"LightGBM Prediction: {lgb_pred:.2f}% (Weight: {ew['lgb']:.2f})")
    else:
        st.info("Forecasting model file not found. Ensure `forecasting_models.joblib` exists in the output directory.")
