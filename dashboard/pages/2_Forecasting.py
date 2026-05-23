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

st.markdown("---")
st.markdown("### 🧮 Interactive Poverty Predictor")
st.markdown("Experiment with socio-economic indicators below to see how they impact the projected poverty rate using our trained Ensemble Model (XGBoost, LightGBM, Random Forest).")

import joblib
import numpy as np

model_path = FORECAST_OUTPUT / "forecasting_models.joblib"
if model_path.exists():
    models = joblib.load(model_path)
    
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
            # Order matches ['poverty_rate_lag1', 'hdi_lag1', 'tpt_lag1', 'tpak_lag1', 'aps_1315_lag1', 'mean_years_schooling_lag1', 'expected_years_schooling_lag1', 'log_poverty_line_lag1']
            X_input = np.array([[poverty_rate, hdi, tpt, tpak, aps_1315, mys, eys, log_poverty_line]])
            
            rf_pred = models['rf_model'].predict(X_input)[0]
            xgb_pred = models['xgb_model'].predict(X_input)[0]
            lgb_pred = models['lgb_model'].predict(X_input)[0]
            ew = models['ensemble_weights']
            
            final_pred = (rf_pred * ew['rf']) + (xgb_pred * ew['xgb']) + (lgb_pred * ew['lgb'])
            
            st.success(f"**Predicted Poverty Rate:** {final_pred:.2f}%")
            
            with st.expander("Show Model Breakdown"):
                st.write(f"Random Forest Prediction: {rf_pred:.2f}% (Weight: {ew['rf']:.2f})")
                st.write(f"XGBoost Prediction: {xgb_pred:.2f}% (Weight: {ew['xgb']:.2f})")
                st.write(f"LightGBM Prediction: {lgb_pred:.2f}% (Weight: {ew['lgb']:.2f})")
else:
    st.info("Forecasting model file not found. Ensure `forecasting_models.joblib` exists in the output directory.")
