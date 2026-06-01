import streamlit as st
import pandas as pd
from pathlib import Path

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">OVERVIEW & SNAPSHOT</div>
        <div class="hero-title">GeoSosEkon Dashboard</div>
        <div class="hero-desc">
            Multimodal Spatial-Temporal Analytics System for Provincial Poverty Dynamics in Indonesia. 
            This system integrates quantitative socioeconomic indicators from BPS with qualitative public sentiment data 
            from Twitter (X) across five advanced analytical layers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-container">
        <h3 style="margin-top: 0; color: #06B6D4; font-family: 'Outfit';">Welcome to the Analytics Platform</h3>
        <p style="color: #94A3B8; line-height: 1.6;">
            This dashboard is the interactive interface of the <b>GeoSosEkon</b> project. The system integrates quantitative socioeconomic data from BPS with qualitative public sentiment data from Twitter, running five independent analytical modules. Use the top navigation menu to switch between:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03);">
                <strong style="color: #06B6D4; display: block; margin-bottom: 4px;">Provincial Typology</strong>
                <span style="font-size: 0.85rem; color: #64748B;">GMM Clustering with PCA & UMAP dimension reduction.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03);">
                <strong style="color: #06B6D4; display: block; margin-bottom: 4px;">Spatial Autocorrelation</strong>
                <span style="font-size: 0.85rem; color: #64748B;">Spatial dependencies using Global Moran's I & Local LISA Maps.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03);">
                <strong style="color: #06B6D4; display: block; margin-bottom: 4px;">Poverty Forecasting</strong>
                <span style="font-size: 0.85rem; color: #64748B;">2025-2026 poverty projections & interactive predictions via XGBoost.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03);">
                <strong style="color: #06B6D4; display: block; margin-bottom: 4px;">Causality & Attribution</strong>
                <span style="font-size: 0.85rem; color: #64748B;">Fixed Effects Panel Regression to identify core poverty drivers.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03);">
                <strong style="color: #06B6D4; display: block; margin-bottom: 4px;">Public Sentiment</strong>
                <span style="font-size: 0.85rem; color: #64748B;">RoBERTa NLP analysis of social media discussions on poverty.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
        except Exception:
            return None

    df = load_data()

    if df is not None:
        st.markdown("<h3 style='margin-bottom: 1.5rem; font-family: \"Outfit\";'>National Poverty Snapshot</h3>", unsafe_allow_html=True)
        
        latest_year = df['year'].max()
        df_latest = df[df['year'] == latest_year]
        
        avg_poverty = df_latest['poverty_rate'].mean()
        highest_prov = df_latest.loc[df_latest['poverty_rate'].idxmax()]
        lowest_prov = df_latest.loc[df_latest['poverty_rate'].idxmin()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-icon-box" style="font-size: 0.8rem; font-weight:700;">AVG</div>
                    <div class="metric-badge">National</div>
                </div>
                <div class="metric-value">{avg_poverty:.2f}%</div>
                <div class="metric-label">Average Rate ({latest_year})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-icon-box" style="color: #EF4444; background: rgba(239, 68, 68, 0.06); border-color: rgba(239, 68, 68, 0.12); font-size: 0.8rem; font-weight:700;">MAX</div>
                    <div class="metric-badge" style="color: #EF4444; background: rgba(239, 68, 68, 0.08); border-color: rgba(239, 68, 68, 0.15)">Highest</div>
                </div>
                <div class="metric-value" style="font-size: 2.25rem;">{highest_prov['poverty_rate']:.2f}%</div>
                <div class="metric-label">{highest_prov['province']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-icon-box" style="color: #10B981; background: rgba(16, 185, 129, 0.06); border-color: rgba(16, 185, 129, 0.12); font-size: 0.8rem; font-weight:700;">MIN</div>
                    <div class="metric-badge">Lowest</div>
                </div>
                <div class="metric-value" style="font-size: 2.25rem;">{lowest_prov['poverty_rate']:.2f}%</div>
                <div class="metric-label">{lowest_prov['province']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="team-container">
        <div class="team-title">Project Team Members</div>
        <ul class="team-list">
            <li class="team-member"><strong>Rizki Piji Fathoni</strong> (24031554029)</li>
            <li class="team-member"><strong>Muhammad Rafi Fahrezi</strong> (24031554100)</li>
            <li class="team-member"><strong>Nazril Ravi Pratama</strong> (2403155129)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: #475569; font-size: 0.85rem; margin-top: 2rem;'>© 2026 GeoSosEkon Project - Data Mining Final Project</p>", unsafe_allow_html=True)
