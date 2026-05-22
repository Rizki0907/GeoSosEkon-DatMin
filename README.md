# Indonesian Poverty Data Mining Project (2021-2026)

This repository contains a comprehensive data mining project analyzing poverty trends, spatial autocorrelation, panel data regression, clustering, and sentiment analysis for Indonesian provinces from 2021 to 2026. The end product is an interactive Streamlit dashboard that visualizes the results.

## Team Members
* Rizki Piji Fathoni (24031554029)
* Muhammad Rafi Fahrezi (24031554100)
* Nazril Ravi Pratama (2403155129)

## Project Architecture

The project is structured into five distinct analytical layers, followed by a unified dashboard:

1. **LAYER 1 - CLUSTERING**
   Applies K-Means clustering with PCA to group Indonesian provinces based on socio-economic indicators (e.g., poverty rate, Human Development Index, unemployment rate, labor force participation rate).
   
2. **LAYER 2 - FORECAST + SHAP**
   Uses XGBoost to forecast poverty rates. It also incorporates SHAP (SHapley Additive exPlanations) to interpret feature importance globally and per cluster.

3. **LAYER 3 - SPATIAL AUTOCORRELATION**
   Utilizes Moran's I and LISA (Local Indicators of Spatial Association) to detect spatial dependencies and geographic clustering of poverty across provinces.

4. **LAYER 4 - FIXED EFFECT PANEL REGRESSION**
   Employs panel data econometric models (Pooled OLS, Fixed Effects, Random Effects) to identify significant socio-economic determinants of poverty over time.

5. **LAYER 5 - ROBERTA SENTIMENT**
   Analyzes Twitter (X) data related to poverty in Indonesia using a RoBERTa-based NLP model to understand public sentiment and engagement metrics.

## Running the Dashboard

To run the interactive dashboard locally:

1. Ensure all dependencies are installed.
2. Navigate to the project root directory.
3. Run the following command:
   ```bash
   streamlit run dashboard/app.py
   ```

## Requirements

The project requires Python 3.12+ and dependencies including:
* pandas, numpy, scikit-learn
* xgboost, shap
* statsmodels, linearmodels
* geopandas, libpysal, esda
* transformers, torch
* plotly, streamlit
