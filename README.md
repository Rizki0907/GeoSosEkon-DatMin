<div align="center">
  <h1>GeoSosEkon: Multimodal Spatial-Temporal Analytics System</h1>
  <h3>Provincial Poverty Dynamics in Indonesia (2021-2026)</h3>

  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geososekon-datmin-gp5uqesskmm5dnqhaxfn2b.streamlit.app/)
  [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<br>

This repository contains the final project for the **Data Mining** course. It presents a comprehensive, end-to-end data mining pipeline analyzing poverty trends across 38 Indonesian provinces from 2021 to 2026. 

By integrating quantitative socio-economic data from **Badan Pusat Statistik (BPS)** with qualitative public sentiment data mined from **Twitter (X)**, this project builds a robust **Multimodal Spatial-Temporal Analytics System** presented via an interactive Streamlit web dashboard.

---

## Team Members
- **Rizki Piji Fathoni** (24031554029)
- **Muhammad Rafi Fahrezi** (24031554100)
- **Nazril Ravi Pratama** (2403155129)

---

## Live Dashboard
The final interactive dashboard is deployed on Streamlit Community Cloud and can be accessed anywhere:

**[Launch GeoSosEkon Dashboard](https://geososekon-datmin-gp5uqesskmm5dnqhaxfn2b.streamlit.app/)**

### Dashboard Features
- **Modern UI/UX**: Built with a sleek Glassmorphism aesthetic, responsive design, and intuitive native sidebar navigation.
- **Interactive Visualizations**: Powered by Plotly to allow zooming, panning, and hovering over complex spatial maps, 3D PCA projections, and sentiment heatmaps.
- **Live Poverty Predictor**: An interactive form that lets you tweak socio-economic variables (HDI, TPT, TPAK, etc.) and predicts the poverty rate instantly using our trained Ensemble model (XGBoost, LightGBM, Random Forest).
- **PDF Report Generator**: Select any province to automatically compile and download a comprehensive 5-layer analytical PDF report built using `fpdf2`.

---

## Project Architecture (The 5 Layers)

The analytical engine behind the dashboard is structured into five distinct Jupyter Notebooks, ensuring a modular and reproducible pipeline:

1. **`LAYER 1` - PROVINCIAL TYPOLOGY (CLUSTERING)**
   Applies Gaussian Mixture Models (GMM) combined with PCA and UMAP dimensionality reduction to accurately group Indonesian provinces based on multi-dimensional socio-economic indicators.
   
2. **`LAYER 2` - FORECASTING & EXPLAINABILITY**
   Uses XGBoost, LightGBM, and Random Forest models to forecast 2025-2026 poverty rates. It incorporates **SHAP** (SHapley Additive exPlanations) to interpret global and cluster-specific feature importance.

3. **`LAYER 3` - SPATIAL AUTOCORRELATION**
   Utilizes Global Moran's I and LISA (Local Indicators of Spatial Association) to detect spatial dependencies, poverty hotspots (High-High), and coldspots (Low-Low) across the Indonesian archipelago.

4. **`LAYER 4` - FIXED EFFECT PANEL REGRESSION**
   Employs panel data econometric models to identify causal socio-economic determinants of poverty over the 4-year temporal period, isolating unobserved provincial heterogeneity.

5. **`LAYER 5` - PUBLIC SENTIMENT (RoBERTa NLP)**
   Analyzes thousands of Twitter (X) tweets related to poverty in Indonesia using a pre-trained IndoBERT/RoBERTa NLP model to extract public sentiment trends and engagement metrics.

---

## Local Installation & Usage

To run the interactive dashboard locally on your own machine:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rizki0907/GeoSosEkon-DatMin.git
   cd GeoSosEkon-DatMin
   ```

2. **Install the Dashboard Dependencies**
   It's highly recommended to use a virtual environment (`venv` or `conda`).
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` contains only the packages necessary to run the Streamlit UI and inference models. If you wish to re-run the Jupyter Notebooks from scratch, you will additionally need PyTorch, Transformers, Geopandas, Libpysal, etc.)*

3. **Run the Streamlit App**
   ```bash
   streamlit run dashboard/app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

<div align="center">
  <p>© 2026 GeoSosEkon Project - Data Mining Final Project</p>
</div>
