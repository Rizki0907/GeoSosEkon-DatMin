import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Causality & SHAP", page_icon="🔍", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_css(file_name):
    try:
        with open(BASE_DIR / "dashboard" / file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css('style.css')

st.markdown("<h1>🔍 Causality & Attribution (Panel Regression & SHAP)</h1>", unsafe_allow_html=True)
st.markdown("---")

LAYER4_DIR = BASE_DIR / "LAYER 4 - FIXED EFFECT PANEL REGRESSION" / "panel_output"
LAYER2_DIR = BASE_DIR / "LAYER 2 - FORECAST + SHAP" / "forecast_output"

st.markdown("## 1. Causality Analysis (Fixed Effect Panel Regression)")
st.markdown("Analyze causal relationships between socio-economic indicators and poverty by controlling for province-specific fixed effects.")

col1, col2 = st.columns(2)
with col1:
    try:
        df_coef = pd.read_csv(LAYER4_DIR / "output_fe_coefficient.csv")
        st.markdown("### Regression Coefficient Table")
        st.dataframe(df_coef, use_container_width=True)
    except Exception as e:
        st.error(f"Coefficient data not found. {e}")

with col2:
    try:
        img_fe = Image.open(LAYER4_DIR / "plot_panel_fe_effects.png")
        st.image(img_fe, use_container_width=True, caption="Individual Fixed Effects per Province")
    except Exception as e:
        st.warning("FE image not found.")

st.markdown("---")

st.markdown("## 2. Feature Attribution (SHAP - Machine Learning Model)")
st.markdown("Explains the contribution of each feature in the poverty forecasting model using the Ensemble Tree algorithm.")

col3, col4 = st.columns(2)
with col3:
    try:
        img_shap_bar = Image.open(LAYER2_DIR / "plot_shap_summary_bar.png")
        st.image(img_shap_bar, use_container_width=True, caption="Global Feature Importance (Mean Absolute SHAP)")
    except Exception as e:
        st.warning(f"SHAP Bar image not found. {e}")

with col4:
    try:
        img_shap_bee = Image.open(LAYER2_DIR / "plot_shap_summary_beeswarm.png")
        st.image(img_shap_bee, use_container_width=True, caption="SHAP Beeswarm Plot (Directional Impact)")
    except Exception as e:
        st.warning(f"SHAP Beeswarm image not found. {e}")
