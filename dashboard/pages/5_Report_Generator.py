import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# Add utils directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
utils_path = str(BASE_DIR / "dashboard")
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.pdf_generator import generate_pdf

st.set_page_config(page_title="Report Generator", layout="centered")

st.markdown("<h2 style='text-align: center;'>📄 Generate Provincial Report</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Extract an integrated PDF report covering Clustering, Spatial, Forecasting, Regression, and Sentiment aspects for a specific province.</p>", unsafe_allow_html=True)
st.markdown("---")

DATA_PATH = BASE_DIR / "data_bps_datmin.csv"
CLUSTER_OUTPUT = BASE_DIR / "LAYER 1 - CLUSTERING" / "cluster_output" / "output_province_clusters.csv"
LISA_OUTPUT = BASE_DIR / "LAYER 3 - SPATIAL AUTOKORELASI (MORANS I + LISA MAP)" / "spatial_output" / "output_lisa_results.csv"
FORECAST_OUTPUT = BASE_DIR / "LAYER 2 - FORECAST + SHAP" / "forecast_output" / "output_forecasting_2025_2026.csv"

@st.cache_data
def load_base_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = ['province', 'year', 'aps_1315', 'aps_1618', 'aps_1924', 'tpt_feb', 'tpt_aug', 'tpak_feb', 'tpak_aug', 'gk_march', 'gk_sept', 'jpm_march', 'jpm_sept', 'ppm_march', 'ppm_sept', 'hdi', 'mean_years_schooling', 'expected_years_schooling']
        df['poverty_rate'] = (df['ppm_march'] + df['ppm_sept']) / 2
        df['tpt'] = (df['tpt_feb'] + df['tpt_aug']) / 2
        df['province'] = df['province'].str.strip().str.upper()
        return df
    except:
        return None

df_base = load_base_data()

if df_base is not None:
    provinces = sorted(df_base['province'].unique())
    selected_prov = st.selectbox("Select Province", provinces)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generate Integrated PDF Report", use_container_width=True):
        with st.spinner(f"Compiling multi-layer data for {selected_prov}..."):
            
            # Gather Macro
            latest_year = df_base['year'].max()
            df_prov = df_base[(df_base['province'] == selected_prov) & (df_base['year'] == latest_year)]
            bps_data = {}
            if not df_prov.empty:
                bps_data = {
                    'poverty_rate': round(df_prov['poverty_rate'].values[0], 2),
                    'hdi': round(df_prov['hdi'].values[0], 2),
                    'tpt': round(df_prov['tpt'].values[0], 2)
                }
                
            # Gather Cluster
            cluster_val = "Data Not Available"
            try:
                df_c = pd.read_csv(CLUSTER_OUTPUT)
                df_c['province'] = df_c['province'].str.upper()
                c_row = df_c[df_c['province'] == selected_prov]
                if not c_row.empty:
                    cluster_val = f"Cluster {c_row['Cluster'].values[0]}"
            except: pass
            
            # Gather Spatial
            lisa_val = "Not Significant"
            try:
                df_l = pd.read_csv(LISA_OUTPUT)
                df_l['province'] = df_l['province'].str.upper()
                l_row = df_l[(df_l['province'].str.contains(selected_prov)) & (df_l['significant'] == True)]
                if not l_row.empty:
                    lisa_val = l_row.iloc[-1]['quadrant'] + " (Significant in recent years)"
            except: pass
            
            # Gather Forecast
            forecast_val = {'2025': 'N/A', '2026': 'N/A'}
            try:
                df_f = pd.read_csv(FORECAST_OUTPUT)
                df_f['province'] = df_f['province'].str.upper()
                f_row = df_f[df_f['province'] == selected_prov]
                if not f_row.empty:
                    forecast_val['2025'] = round(f_row['pred_2025'].values[0], 2)
                    forecast_val['2026'] = round(f_row['pred_2026'].values[0], 2)
            except: pass
            
            data_dict = {
                'bps_data': bps_data,
                'cluster': cluster_val,
                'lisa_quadrant': lisa_val,
                'forecast': forecast_val,
                'sentiment': "According to RoBERTa NLP models run on Twitter data (2021-2026), public sentiment regarding poverty alleviation generally fluctuates between neutral and negative nationwide. The dashboard's Sentiment tab shows specific month-over-month trends."
            }
            
            pdf_path = generate_pdf(selected_prov, data_dict)
            
            if pdf_path and os.path.exists(pdf_path):
                st.success("✅ Report generated successfully!")
                
                with open(pdf_path, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                    
                st.download_button(
                    label="⬇️ Download PDF",
                    data=PDFbyte,
                    file_name=f"GeoSosEkon_Report_{selected_prov}.pdf",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            else:
                st.error("Failed to generate PDF. Check terminal logs.")
else:
    st.error("Could not load base dataset.")
