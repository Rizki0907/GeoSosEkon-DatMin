import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="GeoSosEkon Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

def load_css(file_name):
    try:
        with open(BASE_DIR / "dashboard" / file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css('style.css')

DATA_PATH = BASE_DIR / "data_bps_datmin.csv"

@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv(DATA_PATH)
        df_raw.columns = [
            'province', 'year', 'aps_1315', 'aps_1618', 'aps_1924',
            'tpt_feb', 'tpt_aug', 'tpak_feb', 'tpak_aug',
            'gk_march', 'gk_sept',
            'jpm_march', 'jpm_sept',
            'ppm_march', 'ppm_sept',
            'hdi', 'mean_years_schooling', 'expected_years_schooling'
        ]
        df = df_raw.copy()
        df['poverty_rate'] = (df['ppm_march'] + df['ppm_sept']) / 2
        df['tpt'] = (df['tpt_feb'] + df['tpt_aug']) / 2
        df['tpak'] = (df['tpak_feb'] + df['tpak_aug']) / 2
        df['poverty_line'] = (df['gk_march'] + df['gk_sept']) / 2
        df['province'] = df['province'].str.strip().str.upper()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

st.markdown("<h1 style='text-align: center; color: #2c3e50;'> GeoSosEkon Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>Multimodal Spatial-Temporal Analytics System for Provincial Poverty Dynamics in Indonesia</h3>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="glass-container">
    <h3>Welcome!</h3>
    <p>This dashboard is the interactive interface of the <b>GeoSosEkon</b> project. The system integrates quantitative socioeconomic data from BPS with qualitative public sentiment data from Twitter, running five independent analytical modules.</p>
    <p>Please navigate through the sidebar menu to explore each analytical layer:</p>
    <ul>
        <li><b>Provincial Typology (Clustering)</b></li>
        <li><b>Poverty Hotspot Map (Spatial)</b></li>
        <li><b>Projections & Predictions 2025-2026 (Forecasting)</b></li>
        <li><b>Causality & Attribution (Panel Regression & SHAP)</b></li>
        <li><b>Public Sentiment (RoBERTa Sentiment)</b></li>
    </ul>
</div>
""", unsafe_allow_html=True)

if df is not None:
    st.markdown("### 📊 National Poverty Snapshot (Latest Year)")
    
    latest_year = df['year'].max()
    df_latest = df[df['year'] == latest_year]
    
    avg_poverty = df_latest['poverty_rate'].mean()
    highest_prov = df_latest.loc[df_latest['poverty_rate'].idxmax()]
    lowest_prov = df_latest.loc[df_latest['poverty_rate'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">National Average ({latest_year})</div>
            <div class="metric-value">{avg_poverty:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #e74c3c;">
            <div class="metric-label">Highest ({highest_prov['province']})</div>
            <div class="metric-value">{highest_prov['poverty_rate']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #2ecc71;">
            <div class="metric-label">Lowest ({lowest_prov['province']})</div>
            <div class="metric-value">{lowest_prov['poverty_rate']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 GeoSosEkon Project - Data Mining Final Project")
