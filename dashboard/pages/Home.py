import streamlit as st
import pandas as pd
from pathlib import Path

def show():
    # Page Hero exactly like the reference
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg-grid"></div>
        <div class="hero-glow"></div>
        <div class="hero-eyebrow">Overview & Snapshot</div>
        <h1 class="hero-title">
            GeoSosEkon<br><span class="grad">Dashboard System</span>
        </h1>
        <p class="hero-sub">
            Multimodal Spatial-Temporal Analytics System for Provincial Poverty Dynamics in Indonesia. 
            This system integrates quantitative socioeconomic indicators from BPS with qualitative public sentiment data 
            from Twitter (X) across five advanced analytical layers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-container">
        <h3 style="margin-top: 0; color: #00D4FF; font-family: 'Plus Jakarta Sans'; font-size: 1.25rem;">Analytical Framework</h3>
        <p style="color: #94A3B8; line-height: 1.6; font-size: 0.92rem; margin-bottom: 20px;">
            The system integrates quantitative socioeconomic data from BPS with qualitative public sentiment data from Twitter, running five independent analytical modules. Use the top navigation menu to switch between:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid var(--border);">
                <strong style="color: #00D4FF; display: block; margin-bottom: 4px; font-family: 'Plus Jakarta Sans'; font-size: 0.88rem;">Provincial Typology</strong>
                <span style="font-size: 0.82rem; color: var(--muted); line-height: 1.5; display: block;">GMM Clustering with PCA & UMAP dimension reduction.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid var(--border);">
                <strong style="color: #00D4FF; display: block; margin-bottom: 4px; font-family: 'Plus Jakarta Sans'; font-size: 0.88rem;">Spatial Autocorrelation</strong>
                <span style="font-size: 0.82rem; color: var(--muted); line-height: 1.5; display: block;">Spatial dependencies using Global Moran's I & Local LISA Maps.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid var(--border);">
                <strong style="color: #00D4FF; display: block; margin-bottom: 4px; font-family: 'Plus Jakarta Sans'; font-size: 0.88rem;">Poverty Forecasting</strong>
                <span style="font-size: 0.82rem; color: var(--muted); line-height: 1.5; display: block;">2025-2026 poverty projections & interactive predictions via XGBoost.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid var(--border);">
                <strong style="color: #00D4FF; display: block; margin-bottom: 4px; font-family: 'Plus Jakarta Sans'; font-size: 0.88rem;">Causality & Attribution</strong>
                <span style="font-size: 0.82rem; color: var(--muted); line-height: 1.5; display: block;">Fixed Effects Panel Regression to identify core poverty drivers.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 16px; border-radius: 12px; border: 1px solid var(--border);">
                <strong style="color: #00D4FF; display: block; margin-bottom: 4px; font-family: 'Plus Jakarta Sans'; font-size: 0.88rem;">Public Sentiment</strong>
                <span style="font-size: 0.82rem; color: var(--muted); line-height: 1.5; display: block;">RoBERTa NLP analysis of social media discussions on poverty.</span>
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
        latest_year = df['year'].max()
        df_latest = df[df['year'] == latest_year]
        
        avg_poverty = df_latest['poverty_rate'].mean()
        highest_prov = df_latest.loc[df_latest['poverty_rate'].idxmax()]
        lowest_prov = df_latest.loc[df_latest['poverty_rate'].idxmin()]
        
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title">National Poverty Snapshot ({latest_year})</div>
            <div class="section-line"></div>
        </div>
        
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-badge">National</div>
                <div class="kpi-icon"><i class="fa-solid fa-chart-line"></i></div>
                <div class="kpi-value">{avg_poverty:.2f}%</div>
                <div class="kpi-label">Average Poverty Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background:rgba(239,68,68,0.12); color:#EF4444;"><i class="fa-solid fa-triangle-exclamation"></i></div>
                <div class="kpi-value" style="font-size: 1.85rem;">{highest_prov['poverty_rate']:.2f}%</div>
                <div class="kpi-label">{highest_prov['province']} (Highest)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon" style="background:rgba(16,185,129,0.12); color:#10B981;"><i class="fa-solid fa-circle-check"></i></div>
                <div class="kpi-value" style="font-size: 1.85rem;">{lowest_prov['poverty_rate']:.2f}%</div>
                <div class="kpi-label">{lowest_prov['province']} (Lowest)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-title">Development Team</div>
        <div class="section-line"></div>
    </div>
    <div class="team-grid">
        <div class="team-card">
            <div class="team-avatar">RF</div>
            <div class="team-name">Rizki Piji Fathoni</div>
            <div class="team-id">24031554029</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">MR</div>
            <div class="team-name">Muhammad Rafi Fahrezi</div>
            <div class="team-id">24031554100</div>
        </div>
        <div class="team-card">
            <div class="team-avatar">NR</div>
            <div class="team-name">Nazril Ravi Pratama</div>
            <div class="team-id">2403155129</div>
        </div>
    </div>
    <p style='text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 3rem; font-family: var(--font-mono);'>
        © 2026 GeoSosEkon Project &bull; Data Mining Final Project
    </p>
    """, unsafe_allow_html=True)
