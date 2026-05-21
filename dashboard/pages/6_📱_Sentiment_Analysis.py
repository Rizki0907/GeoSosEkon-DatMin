import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Sentiment Analysis", page_icon="📱", layout="wide")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css('style.css')

st.markdown("<h1>📱 Sentimen Publik (Twitter IndoRoBERTa)</h1>", unsafe_allow_html=True)
st.markdown("---")

OUTPUT_DIR = Path("../LAYER 5 - ROBERTA SENTIMENT/sentiment_output")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Korelasi Sentimen vs Tingkat Kemiskinan")
    try:
        img_corr = Image.open(OUTPUT_DIR / "plot_sentiment_poverty_correlation.png")
        st.image(img_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Image not found. {e}")

with col2:
    st.markdown("### Peta Sentimen per Provinsi")
    try:
        img_map = Image.open(OUTPUT_DIR / "plot_sentiment_map.png")
        st.image(img_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Image not found. {e}")

st.markdown("---")
st.markdown("### Ringkasan Sentimen per Provinsi")
try:
    df_sent = pd.read_csv(OUTPUT_DIR / "output_sentiment_by_province.csv")
    st.dataframe(df_sent, use_container_width=True)
except Exception as e:
    st.error("CSV Data not found.")
