import streamlit as st
from pathlib import Path
import sys
import importlib

# Ensure dashboard path is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))





# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="GeoSosEkon Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import traceback

# 2. Dynamic Import of Sub-modules (handles numerical filenames cleanly)
try:
    Home = importlib.import_module("pages.Home")
    Clustering = importlib.import_module("pages.0_Clustering")
    Spatial_Analysis = importlib.import_module("pages.1_Spatial_Analysis")
    Forecasting = importlib.import_module("pages.2_Forecasting")
    Causality_and_SHAP = importlib.import_module("pages.3_Causality_and_SHAP")
    Sentiment_Analysis = importlib.import_module("pages.4_Sentiment_Analysis")
    Report_Generator = importlib.import_module("pages.5_Report_Generator")
except Exception as e:
    st.error(f"Error loading pages: {e}")
    st.text(traceback.format_exc())
    st.stop()

# 3. Load CSS Stylesheet
def load_css(file_name):
    try:
        with open(BASE_DIR / file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass



load_css('style.css')

# 4. Routing via Query Parameters
query_params = st.query_params
active_page = query_params.get("page", "overview")

# Define active CSS highlights
active_overview = "active" if active_page == "overview" else ""
active_typology = "active" if active_page == "typology" else ""
active_spatial = "active" if active_page == "spatial" else ""
active_forecasting = "active" if active_page == "forecasting" else ""
active_causality = "active" if active_page == "causality" else ""
active_sentiment = "active" if active_page == "sentiment" else ""
active_report = "active" if active_page == "report" else ""

# 5. Render Premium Top Navigation Bar
st.markdown(f"""
<div class="nav-container">
    <div class="nav-logo-section">
        <span class="nav-logo-text">GEOSOSEKON</span>
    </div>
    <div class="nav-menu-items">
        <a href="?page=overview" class="nav-item {active_overview}" target="_self">Overview</a>
        <a href="?page=typology" class="nav-item {active_typology}" target="_self">Typology</a>
        <a href="?page=spatial" class="nav-item {active_spatial}" target="_self">Spatial</a>
        <a href="?page=forecasting" class="nav-item {active_forecasting}" target="_self">Forecasting</a>
        <a href="?page=causality" class="nav-item {active_causality}" target="_self">Causality</a>
        <a href="?page=sentiment" class="nav-item {active_sentiment}" target="_self">Sentiment</a>
        <a href="?page=report" class="nav-item {active_report}" target="_self">Report</a>
    </div>
    <div class="nav-meta-info">
        Final Project 2026
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Render the Active View Component
if active_page == "overview":
    Home.show()
elif active_page == "typology":
    Clustering.show()
elif active_page == "spatial":
    Spatial_Analysis.show()
elif active_page == "forecasting":
    Forecasting.show()
elif active_page == "causality":
    Causality_and_SHAP.show()
elif active_page == "sentiment":
    Sentiment_Analysis.show()
elif active_page == "report":
    Report_Generator.show()
else:
    Home.show()
