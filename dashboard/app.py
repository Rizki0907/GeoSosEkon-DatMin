import streamlit as st
from pathlib import Path
import sys
import importlib
import traceback
import re

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

# 2. Dynamic Import of Sub-modules
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

# 3. Load CSS Stylesheet & CDNs (Fonts + FontAwesome)
def load_and_inject_assets():
    try:
        with open(BASE_DIR / "style.css") as f:
            css_content = f.read()
        
        # Minify CSS slightly (replace multiple linebreaks)
        clean_css = re.sub(r'\n{2,}', '\n', css_content.strip())
        
        # Inject Google Fonts + FontAwesome + CSS
        st.markdown(
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">'
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">'
            f'<style>{clean_css}</style>',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

load_and_inject_assets()

# 4. Navigation Configurations
NAV_ITEMS = {
    "Overview": "fa-chart-pie",
    "Typology": "fa-circle-nodes",
    "Spatial": "fa-earth-asia",
    "Forecasting": "fa-chart-line",
    "Causality": "fa-link",
    "Sentiment": "fa-comments",
    "Report": "fa-file-pdf"
}

PAGE_SLUGS = {
    "Overview": "overview",
    "Typology": "typology",
    "Spatial": "spatial",
    "Forecasting": "forecasting",
    "Causality": "causality",
    "Sentiment": "sentiment",
    "Report": "report"
}
SLUG_TO_PAGE = {v: k for k, v in PAGE_SLUGS.items()}

# 5. Routing via Query Parameters (using target="_top" for iframe escapes)
slug = st.query_params.get("page", "overview")
if isinstance(slug, list):
    slug = slug[0] if slug else "overview"
page = SLUG_TO_PAGE.get(str(slug).lower(), "Overview")

# Render Sticky Navigation Bar
nav_html = '<div class="nav-wrapper"><div class="nav-logo"><div class="nav-dot"></div><span>GEO</span>SOSEKON</div><div class="nav-links">'
for item, icon in NAV_ITEMS.items():
    is_active = page == item
    active_cls = ' class="active"' if is_active else ''
    nav_html += f'<a href="?page={PAGE_SLUGS[item]}" target="_top"{active_cls}><i class="fa-solid {icon}" style="font-size:0.8rem;"></i> {item}</a>'
nav_html += '</div><div class="nav-team">DatMin Project &bull; 2026</div></div>'
st.markdown(nav_html, unsafe_allow_html=True)

# 6. Render the Active View Component (with page wrapper margin)
st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

if page == "Overview":
    Home.show()
elif page == "Typology":
    Clustering.show()
elif page == "Spatial":
    Spatial_Analysis.show()
elif page == "Forecasting":
    Forecasting.show()
elif page == "Causality":
    Causality_and_SHAP.show()
elif page == "Sentiment":
    Sentiment_Analysis.show()
elif page == "Report":
    Report_Generator.show()
else:
    Home.show()

st.markdown('</div>', unsafe_allow_html=True)
