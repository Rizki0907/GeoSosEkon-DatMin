import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import joblib

st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_css(file_name):
    try:
        with open(BASE_DIR / "dashboard" / file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css('style.css')

st.markdown("<h1>📈 Poverty Projections 2025-2026</h1>", unsafe_allow_html=True)
st.markdown("---")

OUTPUT_DIR = BASE_DIR / "LAYER 2 - FORECAST + SHAP" / "forecast_output"

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Historical & Projected National Average Trend")
    try:
        img_trend = Image.open(OUTPUT_DIR / "plot_forecast_combined_trend.png")
        st.image(img_trend, use_container_width=True)
    except Exception as e:
        st.error(f"Image not found. {e}")

with col2:
    st.markdown("### Spaghetti Plot (Per Province)")
    try:
        img_spaghetti = Image.open(OUTPUT_DIR / "plot_forecast_spaghetti.png")
        st.image(img_spaghetti, use_container_width=True)
    except Exception as e:
        st.warning("Image not found.")

st.markdown("---")
st.markdown("### Interactive What-If Prediction")
st.markdown("Use the sliders below to modify provincial socioeconomic indicators and observe the resulting poverty rate predicted by the Ensemble Model (Random Forest + XGBoost + LightGBM).")

try:
    model_data = joblib.load(OUTPUT_DIR / "forecasting_models.joblib")
    rf_model = model_data['rf_model']
    xgb_model = model_data['xgb_model']
    lgb_model = model_data['lgb_model']
    weights = model_data['ensemble_weights']
    features = model_data['features']
    
    st.success("✅ Models loaded successfully!")
    
    input_cols = st.columns(4)
    input_data = {}
    
    for i, feat in enumerate(features):
        with input_cols[i % 4]:
            val = st.number_input(f"{feat}", value=10.0, step=0.1)
            input_data[feat] = [val]
            
    if st.button("Predict Poverty Rate"):
        df_input = pd.DataFrame(input_data)
        pred_rf = rf_model.predict(df_input)[0]
        pred_xgb = xgb_model.predict(df_input)[0]
        pred_lgb = lgb_model.predict(df_input)[0]
        
        pred_ens = (weights['rf'] * pred_rf) + (weights['xgb'] * pred_xgb) + (weights['lgb'] * pred_lgb)
        
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #f39c12; width: 50%; margin: 0 auto;">
            <div class="metric-label">Predicted Poverty Rate</div>
            <div class="metric-value">{pred_ens:.2f}%</div>
            <div style="font-size:0.8rem; color:#7f8c8d;">Ensemble Weight: RF ({weights['rf']:.2f}) | XGB ({weights['xgb']:.2f}) | LGB ({weights['lgb']:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
        
except Exception as e:
    st.error(f"Models not found or failed to load. Run Layer 2 to generate joblib file. Error: {e}")
