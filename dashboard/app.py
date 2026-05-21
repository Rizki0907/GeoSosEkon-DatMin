import streamlit as st
import pandas as pd
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GeoSosEkon Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css('style.css')
except FileNotFoundError:
    pass

# --- CONSTANTS ---
DATA_PATH = Path("../data_bps_datmin.csv")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH)
        df_raw.columns = [
            'provinsi', 'tahun', 'aps_1315', 'aps_1618', 'aps_1924',
            'tpt_feb', 'tpt_agu', 'tpak_feb', 'tpak_agu',
            'gk_maret', 'gk_sept',
            'jpm_maret', 'jpm_sept',
            'ppm_maret', 'ppm_sept',
            'ipm', 'rls', 'hls'
        ]
        df = df_raw.copy()
        df['y_kemiskinan'] = (df['ppm_maret'] + df['ppm_sept']) / 2
        df['tpt'] = (df['tpt_feb'] + df['tpt_agu']) / 2
        df['tpak'] = (df['tpak_feb'] + df['tpak_agu']) / 2
        df['gk'] = (df['gk_maret'] + df['gk_sept']) / 2
        df['provinsi'] = df['provinsi'].str.strip().str.upper()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# --- MAIN CONTENT ---
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🌍 GeoSosEkon Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>Sistem Analitik Spasial-Temporal Dinamika Kemiskinan Provinsi di Indonesia</h3>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="glass-container">
    <h3>Selamat Datang!</h3>
    <p>Dashboard ini merupakan antarmuka interaktif dari proyek <b>GeoSosEkon</b>. Sistem ini menggabungkan data sosioekonomi kuantitatif dari BPS dengan data sentimen publik kualitatif dari Twitter, menjalankan lima modul analitik independen.</p>
    <p>Silakan navigasikan menu di sidebar sebelah kiri untuk mengeksplorasi masing-masing lapisan analitik:</p>
    <ul>
        <li><b>Tipologi Provinsi (Clustering)</b> - Segera Hadir</li>
        <li><b>Peta Hotspot Kemiskinan (Spatial)</b></li>
        <li><b>Proyeksi & Prediksi 2025-2026 (Forecasting)</b></li>
        <li><b>Kausalitas & Atribusi (Panel Regression & SHAP)</b></li>
        <li><b>Sentimen Publik (RoBERTa Sentiment)</b></li>
    </ul>
</div>
""", unsafe_allow_html=True)

if df is not None:
    st.markdown("### 📊 Snapshot Kemiskinan Nasional (Tahun Terakhir)")
    
    latest_year = df['tahun'].max()
    df_latest = df[df['tahun'] == latest_year]
    
    avg_poverty = df_latest['y_kemiskinan'].mean()
    highest_prov = df_latest.loc[df_latest['y_kemiskinan'].idxmax()]
    lowest_prov = df_latest.loc[df_latest['y_kemiskinan'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Nasional ({latest_year})</div>
            <div class="metric-value">{avg_poverty:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #e74c3c;">
            <div class="metric-label">Tertinggi ({highest_prov['provinsi']})</div>
            <div class="metric-value">{highest_prov['y_kemiskinan']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #2ecc71;">
            <div class="metric-label">Terendah ({lowest_prov['provinsi']})</div>
            <div class="metric-value">{lowest_prov['y_kemiskinan']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 GeoSosEkon Project - Data Mining Final Project")
