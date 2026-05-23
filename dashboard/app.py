import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="GeoSosEkon Dashboard",
    page_icon="🌍",
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

# Streamlit Native Navigation (Streamlit >= 1.36)
pages = {
    "🏠 Overview": [
        st.Page("pages/Home.py", title="Dashboard Home", icon="🏠"),
    ],
    "📊 Analytics Engine": [
        st.Page("pages/0_Clustering.py", title="Provincial Typology", icon="🎯"),
        st.Page("pages/1_Spatial_Analysis.py", title="Spatial Autocorrelation", icon="🗺️"),
        st.Page("pages/2_Forecasting.py", title="Poverty Forecasting", icon="📈"),
        st.Page("pages/3_Causality_and_SHAP.py", title="Causality (Panel Data)", icon="🔗"),
        st.Page("pages/4_Sentiment_Analysis.py", title="Public Sentiment", icon="💬"),
    ],
    "📑 Reporting": [
        st.Page("pages/5_Report_Generator.py", title="Generate Provincial Report", icon="📄"),
    ]
}

pg = st.navigation(pages)

# Custom Sidebar Styling Override (Hide default Streamlit logo, etc if needed)
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {padding-top: 2rem;}
        [data-testid="stSidebar"] {min-width: 280px; max-width: 280px;}
    </style>
""", unsafe_allow_html=True)

# Run the selected page
pg.run()
