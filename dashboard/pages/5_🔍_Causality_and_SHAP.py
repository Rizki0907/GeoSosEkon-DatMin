import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Causality & SHAP", page_icon="🔍", layout="wide")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css('style.css')

st.markdown("<h1>🔍 Kausalitas & Atribusi (Panel Regression & SHAP)</h1>", unsafe_allow_html=True)
st.markdown("---")

LAYER4_DIR = Path("../LAYER 4 - FIXED EFFECT PANEL REGRESSION/panel_output")
LAYER2_DIR = Path("../LAYER 2 - FORECAST + SHAP/forecast_output")

st.markdown("## 1. Analisis Kausalitas (Fixed Effect Panel Regression)")
st.markdown("Mengetahui hubungan sebab-akibat antara indikator sosio-ekonomi dan kemiskinan dengan mengontrol efek spesifik tiap provinsi.")

col1, col2 = st.columns(2)
with col1:
    try:
        df_coef = pd.read_csv(LAYER4_DIR / "output_fe_coefficient.csv")
        st.markdown("### Tabel Koefisien Regresi")
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

st.markdown("## 2. Atribusi Fitur (SHAP - Model Machine Learning)")
st.markdown("Menjelaskan seberapa besar kontribusi masing-masing fitur dalam model *forecasting* kemiskinan yang menggunakan algoritma Ensemble Tree.")

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
