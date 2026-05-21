# GeoSosEkon: Multimodal Spatial-Temporal Poverty Analytics

**GeoSosEkon** is a multimodal integrated spatial-temporal analytics system designed to model and project provincial poverty dynamics in Indonesia (2021 - 2026).

This system was developed as a **Data Mining Final Project**, combining quantitative socioeconomic data from Statistics Indonesia (BPS) with qualitative public sentiment data (Twitter) using five comprehensive analytical modules.

---

## Development Team

This project was developed collaboratively by:

1. **Rizki Piji Fathoni** (24031554029)
2. **Muhammad Rafi Fahrezi** (24031554100)
3. **Nazril Ravi Pratama** (2403155129)

---

## System Architecture (5 Analytical Layers)

The project is divided into 5 independent but complementary analytical layers:

### 1. Provincial Typology (Clustering) - *In Progress*
- Performing dimensionality reduction comparing **PCA vs UMAP**.
- Clustering provincial data using **Gaussian Mixture Model (GMM)** to form socioeconomic typologies.

### 2. Poverty Projections (Forecasting) & SHAP
- Utilizing an **Ensemble Machine Learning** approach (Random Forest + XGBoost + LightGBM) optimized with Inverse-RMSE weighting.
- Projecting poverty rates for the years **2025 and 2026** based on recursive autoregressive panel data.
- Applying **SHAP (SHapley Additive exPlanations)** for global feature attribution to understand which indicators predominantly drive the model predictions.

### 3. Spatial Autocorrelation (Spatial Analysis)
- Analyzing regional dependencies using **Moran's I** (Global) and **LISA** (Local Indicators of Spatial Association).
- Identifying poverty **Hotspots** (High-High) and **Coldspots** (Low-Low) across Indonesia.

### 4. Causal Inference (Panel Regression)
- Running an econometric **Two-Way Fixed Effect Panel Regression** model (controlling for province and time-specific fixed effects).
- Drawing causal inference between human development indicators (HDI, Unemployment Rate, Labor Force Participation Rate, etc.) and actual poverty rates.

### 5. Public Sentiment Analysis (Twitter IndoRoBERTa)
- Analyzing public opinion on Twitter regarding social and economic issues.
- Utilizing a pre-trained **IndoRoBERTa Transformer** model for text classification.
- Calculating the correlation between aggregate yearly negative/positive sentiment and actual poverty percentage.

---

## Interactive Dashboard

This project is equipped with an interactive **Streamlit Dashboard** (located in the `dashboard/` folder) acting as the primary Business Intelligence Tool.

Through this dashboard, you can:
- View the latest poverty summaries and statistics.
- Perform real-time **What-If Analysis** to predict poverty rates if an indicator is altered (using dynamically loaded joblib ML models).
- Explore LISA maps, regression outputs, and SHAP attribution charts.

### How to Run the Dashboard:
1. Clone this repository to your local machine.
2. Open a terminal inside the project directory.
3. Run the following command:
   ```bash
   streamlit run dashboard/app.py
   ```
4. The dashboard will automatically open in your browser (defaulting to `localhost:8501`).

---
*Copyright 2026 GeoSosEkon Project*
