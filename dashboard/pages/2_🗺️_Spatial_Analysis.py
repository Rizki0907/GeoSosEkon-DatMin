import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Spatial Analysis", page_icon="🗺️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_css(file_name):
    try:
        with open(BASE_DIR / "dashboard" / file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css('style.css')

st.markdown("<h1>🗺️ Spatial Autocorrelation (Moran's I & LISA)</h1>", unsafe_allow_html=True)
st.markdown("---")

OUTPUT_DIR = BASE_DIR / "LAYER 3 - SPATIAL AUTOKORELASI (MORANS I + LISA MAP)" / "spatial_output"

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### LISA Map (Local Indicators of Spatial Association)")
    try:
        lisa_map = Image.open(OUTPUT_DIR / "plot_lisa_map.png")
        st.image(lisa_map, use_container_width=True)
    except Exception as e:
        st.error(f"Image not found. Run Layer 3 first. {e}")

with col2:
    st.markdown("### Moran's Scatter Plot")
    try:
        moran_scatter = Image.open(OUTPUT_DIR / "plot_moran_scatter.png")
        st.image(moran_scatter, use_container_width=True)
    except Exception as e:
        st.warning("Image not found.")

st.markdown("---")
st.markdown("### Poverty Hotspot & Coldspot Data")
try:
    df_lisa = pd.read_csv(OUTPUT_DIR / "output_lisa_results.csv")
    st.dataframe(df_lisa, use_container_width=True)
except Exception as e:
    st.error("CSV Data not found.")
