# 🌍 GeoSosEkon: Multimodal Spatial-Temporal Poverty Analytics

**GeoSosEkon** adalah sebuah sistem analitik spasial-temporal terintegrasi multimodal yang dirancang untuk memodelkan dan memproyeksikan dinamika tingkat kemiskinan tingkat provinsi di Indonesia (2021 - 2026). 

Sistem ini dikembangkan sebagai **Proyek Akhir Mata Kuliah Data Mining** dengan menggabungkan data sosio-ekonomi kuantitatif dari Badan Pusat Statistik (BPS) dengan data kualitatif sentimen publik (Twitter) menggunakan lima modul analitik yang komprehensif.

---

## 👥 Tim Pengembang

Proyek ini dikembangkan secara kolaboratif oleh:

1. **Rizki Piji Fathoni** (24031554029)
2. **Muhammad Rafi Fahrezi** (24031554100)
3. **Nazril Ravi Pratama** (2403155129)

---

## 🏗️ Arsitektur Sistem (5 Layer Analitik)

Proyek ini dibagi ke dalam 5 lapisan (*layer*) analisis independen yang saling melengkapi:

### 1. Tipologi Provinsi (Clustering) - *In Progress*
- Melakukan reduksi dimensi membandingkan **PCA vs UMAP**.
- Pengelompokan data provinsi menggunakan **Gaussian Mixture Model (GMM)** untuk membentuk tipologi sosio-ekonomi wilayah.

### 2. Proyeksi Kemiskinan (Forecasting) & SHAP
- Menggunakan pendekatan **Ensemble Machine Learning (Random Forest + XGBoost + LightGBM)** dengan optimasi pembobotan Inverse-RMSE.
- Melakukan proyeksi (*forecasting*) tingkat kemiskinan untuk tahun **2025 dan 2026** berdasarkan data panel *recursive autoregressive*.
- Menerapkan **SHAP (SHapley Additive exPlanations)** untuk atribusi fitur global guna memahami indikator apa yang paling krusial mendrive prediksi model.

### 3. Autokorelasi Spasial (Spatial Analysis)
- Menganalisis ketergantungan wilayah menggunakan **Moran's I** (Global) dan **LISA** (Local Indicators of Spatial Association).
- Mengidentifikasi pemetaan daerah *Hotspot* (Tinggi-Tinggi) dan *Coldspot* (Rendah-Rendah) kemiskinan di Indonesia.

### 4. Inferensi Kausal (Panel Regression)
- Menjalankan model ekonometrika **Two-Way Fixed Effect Panel Regression** (mengontrol efek spesifik individu provinsi dan waktu).
- Tujuannya adalah untuk menarik kesimpulan *causal inference* antara indikator pembangunan manusia (IPM, TPT, TPAK, dll) terhadap tingkat kemiskinan aktual.

### 5. Analisis Sentimen Publik (Twitter IndoRoBERTa)
- Menganalisis opini masyarakat di Twitter mengenai isu sosial dan ekonomi.
- Menggunakan pre-trained model **IndoRoBERTa Transformer** untuk mengklasifikasi teks.
- Menghitung korelasi agregat sentimen negatif/positif per tahun dengan persentase tingkat kemiskinan rill.

---

## 💻 Dashboard Interaktif

Proyek ini dilengkapi dengan **Dashboard Streamlit** (berada di dalam folder `dashboard/`) yang berfungsi sebagai *Business Intelligence Tool*. 

Melalui dashboard ini, Anda dapat:
- Melihat ringkasan dan statistik kemiskinan terbaru.
- Melakukan **What-If Analysis** secara *real-time* untuk memprediksi tingkat kemiskinan jika suatu indikator diubah (menggunakan *joblib loaded ML models*).
- Mengeksplorasi peta LISA, hasil regresi, dan grafik atribusi SHAP.

### Cara Menjalankan Dashboard:
1. *Clone* repository ini ke lokal mesin Anda.
2. Buka terminal di dalam folder proyek.
3. Jalankan perintah berikut:
   ```bash
   streamlit run dashboard/app.py
   ```
4. Dashboard akan terbuka di browser Anda (standar di `localhost:8501`).

---
*© 2026 GeoSosEkon Project*
