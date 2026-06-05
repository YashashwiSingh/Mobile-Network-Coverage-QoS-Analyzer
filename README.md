<div align="center">

# 📶 Mobile Network Coverage & QoS Analyzer
### *Full-Stack · Real-Time Dashboard · ML-Powered · Interactive Coverage Maps*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boost-02569B?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Folium](https://img.shields.io/badge/Folium-Maps-77B829?style=for-the-badge)](https://python-visualization.github.io/folium/)

<br/>

> An end-to-end platform for collecting, analyzing, and visualizing **mobile network signal metrics** across geographic localities.  
> Combines a Python analytics engine with a browser-based React dashboard, ML throughput prediction, interactive Folium coverage maps, and automated PDF reporting.

<br/>

| 📍 Coverage Areas | 📊 Measurements | 🌐 Network Types | 🔬 Measurement Devices |
|:-----------------:|:---------------:|:----------------:|:----------------------:|
| **Multi-locality** | **16,830+ rows** | **3G / 4G / LTE / 5G** | **BB60C · srsRAN · BladeRFxA9** |

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [Python Backend](#-python-backend-wireless_mobile_comms_py)
  - [MobileNetworkAnalyzer Class](#mobilenetworkanalyzer-class)
  - [LightGBM Throughput Prediction](#lightgbm-throughput-prediction)
  - [Gradio Web Interface](#gradio-web-interface)
- [React Frontend Dashboard](#-react-frontend-dashboard-appjs--indexhtml)
- [Coverage Maps](#-coverage-maps)
- [QoS Scoring System](#-qos-scoring-system)
- [ML Models](#-ml-models)
- [Output Files](#-output-files)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🔬 Overview

This project provides a **complete mobile network analysis pipeline** — from raw CSV signal measurements to interactive dashboards, geographic heatmaps, ML-based performance prediction, and auto-generated PDF reports.

### What it Analyzes

| Metric | Description |
|--------|-------------|
| **Signal Strength (dBm)** | Raw radio power level from the device |
| **Signal Quality (%)** | Connection quality as a percentage |
| **Data Throughput (Mbps)** | Actual data transfer speed |
| **Latency (ms)** | Round-trip network response time |
| **BB60C / srsRAN / BladeRFxA9** | Cross-instrument signal measurements for validation |

### Key Capabilities

- ✅ Real dataset — **16,830 measurements** across localities in **Patna, India** (lat ~25.5°N, lon ~85.1°E)
- ✅ **Locality-level coverage analysis** — Downtown, Suburban, Rural, Industrial, Residential
- ✅ **Temporal analysis** — hourly and daily QoS trends
- ✅ **Instrument comparison** — inter-device measurement correlation (BB60C vs srsRAN vs BladeRFxA9)
- ✅ **LightGBM** throughput prediction with GridSearchCV tuning
- ✅ **Random Forest** classifier for signal quality labeling
- ✅ **Interactive Folium coverage maps** + Plotly signal heatmaps
- ✅ **React + Chart.js dashboard** with live CSV upload and in-browser ML
- ✅ **Gradio UI** with PDF report generation via ReportLab

---

## 📦 Dataset

**File:** `signal_metrics.csv` — **16,830 rows**, **13 columns**

```
Timestamp, Locality, Latitude, Longitude,
Signal Strength (dBm), Signal Quality (%), Data Throughput (Mbps),
Latency (ms), Network Type,
BB60C Measurement (dBm), srsRAN Measurement (dBm), BladeRFxA9 Measurement (dBm),
signal_label
```

**Geographic scope:** Localities in and around **Patna, Bihar, India**

| Locality Examples | Network Types |
|-------------------|---------------|
| Anisabad, Fraser Road, Boring Canal Road, Danapur, and more | 3G · 4G · LTE · 5G |

### Data Cleaning Pipeline

```python
# Key cleaning steps applied
dfc = df.drop_duplicates()
dfc[lat_col] = pd.to_numeric(dfc[lat_col], errors='coerce')
dfc = dfc[(dfc[lat_col].abs() <= 90) & (dfc[lon_col].abs() <= 180)]  # geo-validation
for c in metric_cols:
    dfc[c] = dfc[c].fillna(dfc[c].median())   # median imputation
dfc[timestamp_col] = dfc[timestamp_col].fillna(method='ffill')       # forward-fill timestamps

# Signal Quality anomaly fix — replace 0s with random valid values
df['Signal Quality (%)'] = df['Signal Quality (%)'].apply(
    lambda x: np.random.rand() * 100 if x == 0 else x
)
```

---

## 🏗 System Architecture

```
signal_metrics.csv
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│            Python Backend (wireless_mobile_comms_.py)   │
│                                                         │
│  ┌──────────────────────┐   ┌──────────────────────┐   │
│  │  MobileNetworkAnalyzer│   │  LightGBM Pipeline   │   │
│  │  ├─ load_data()       │   │  ├─ Feature Eng.     │   │
│  │  ├─ preprocess_data() │   │  ├─ GridSearchCV     │   │
│  │  ├─ analyze_coverage()│   │  └─ Throughput Pred. │   │
│  │  ├─ analyze_qos()     │   └──────────────────────┘   │
│  │  ├─ temporal_analysis()│                              │
│  │  ├─ predictive_model()│   ┌──────────────────────┐   │
│  │  ├─ coverage_map()    │   │  Gradio Interface    │   │
│  │  └─ generate_report() │   │  ├─ Data Preview     │   │
│  └──────────────────────┘   │  ├─ Visualization     │   │
│                              │  ├─ PDF Report Gen.  │   │
│                              │  └─ CSV Upload       │   │
│                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────┐       ┌───────────────────────┐
│  coverage_map.html│       │  network_report.pdf   │
│  (Folium +        │       │  (ReportLab)          │
│   Plotly heatmap) │       └───────────────────────┘
└──────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│            React Frontend (index.html + app.js)         │
│  ├─ CSV file upload + real-time parsing                 │
│  ├─ Filter panel (locality / network type / thresholds) │
│  ├─ Chart.js: Signal, Throughput, Latency, QoS charts   │
│  ├─ Summary stat cards                                  │
│  └─ In-browser Linear Regression (SimpleMLPredictor)    │
└─────────────────────────────────────────────────────────┘
```

---

## 🐍 Python Backend (`wireless_mobile_comms_.py`)

### `MobileNetworkAnalyzer` Class

The core analysis engine — a single class with a full pipeline:

#### Methods Overview

| Method | Description |
|--------|-------------|
| `load_data(file_path, sample_data)` | Load CSV or generate 1000-sample synthetic data with realistic locality distributions |
| `preprocess_data()` | Parse timestamps, engineer Coverage Quality Score & QoS Score, compute avg instrument measurement |
| `analyze_coverage_statistics()` | Group stats by locality and network type (signal strength, quality, QoS) |
| `analyze_qos_performance()` | Categorize QoS (Excellent/Good/Fair/Poor), compute correlation matrix |
| `instrument_comparison_analysis()` | Inter-device diff columns + correlation matrix for BB60C / srsRAN / BladeRFxA9 |
| `temporal_analysis()` | Hourly and day-of-week QoS/throughput/latency aggregation |
| `create_visualizations()` | 12-panel Matplotlib figure (box plots, histograms, violin plots, heatmaps, scatter plots, pie chart) |
| `create_coverage_map()` | Folium HeatMap + locality marker pins with QoS-coded colors |
| `predictive_modeling()` | Random Forest for throughput and QoS prediction with feature importance |
| `generate_recommendations()` | Rule-based actionable insights (coverage gaps, latency, capacity, peak hours, 5G upgrade) |
| `export_results()` | Save `processed_network_data.csv` + `network_analysis_summary.txt` |
| `run_complete_analysis()` | One-call full pipeline execution |

#### Engineered Features

```python
# Coverage Quality Score: combines signal strength and quality
Coverage_Quality_Score = (
    (Signal_Strength_dBm + 120) / 90 * 50 +   # signal strength component
    Signal_Quality_pct / 100 * 50              # signal quality component
)

# QoS Score: throughput + latency + quality composite
QoS_Score = (
    Data_Throughput_Mbps / 100 * 40 +
    (100 - Latency_ms) / 100 * 30 +
    Signal_Quality_pct / 100 * 30
)

# Avg Instrument Measurement
Avg_Instrument = mean(BB60C, srsRAN, BladeRFxA9)
```

#### QoS Categories

| Score Range | Category |
|-------------|----------|
| ≥ 75 | 🟢 Excellent |
| 60 – 74 | 🟡 Good |
| 45 – 59 | 🟠 Fair |
| < 45 | 🔴 Poor |

---

### LightGBM Throughput Prediction

A second, standalone pipeline using **LightGBM + GridSearchCV** for high-accuracy throughput regression:

#### Feature Engineering

```python
df["Device_Avg_dBm"]        = mean(BB60C, srsRAN, BladeRFxA9)
df["Signal_Latency_Ratio"]  = Signal_Strength / (Latency + 1)
df["Quality_Signal"]        = Signal_Quality * Signal_Strength
df["Signal_Strength_Sq"]    = Signal_Strength²
df["Latency_Sq"]            = Latency²
```

#### Model Configuration

```python
param_grid = {
    "num_leaves":    [31, 50, 80],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators":  [200, 500, 800],
    "max_depth":     [6, 10, -1]
}

grid = GridSearchCV(lgb.LGBMRegressor(), param_grid, cv=3, scoring="r2", n_jobs=-1)
```

**Target:** `Data Throughput (Mbps)` — regression task with 80/20 train-test split.

---

### Gradio Web Interface

A 4-tab interactive UI built with **Gradio Blocks**:

| Tab | Features |
|-----|---------|
| 📊 **Data Preview** | Slider to control row count, live dataset shape display |
| 📈 **Visualization** | One-click generation of histogram, line graph, heatmap, and scatter plot |
| ℹ️ **About Model** | Process flow, key metrics, and technology documentation |
| 📋 **Generate Report** | CSV upload, custom notes input, PDF download via ReportLab |

**Launch mode:** `demo.launch(share=True)` for Google Colab public URL sharing.

---

## ⚛️ React Frontend Dashboard (`app.js` + `index.html`)

A **standalone browser app** — no build tools required. Opens directly from `index.html`.

### Features

- **CSV drag-and-drop / file upload** — parses all 12 columns client-side
- **Sample data generator** — 50 synthetic records across 8 localities (Downtown, Airport, Hospital, Stadium, etc.) with randomized coordinates near Bengaluru (~12.97°N, 77.59°E)
- **Filter panel** — filter by locality, network type, min signal strength, max latency
- **Chart.js visualizations** — signal trend, throughput, latency, QoS charts rendered in real-time
- **Summary stat cards** — avg signal, throughput, latency, QoS score
- **Quality badges** — Excellent / Good / Fair / Poor color-coded per data row
- **In-browser ML** — `SimpleMLPredictor` class implements full linear regression from scratch (normal equation: `β = (XᵀX)⁻¹Xᵀy`) for signal quality prediction

### `SimpleMLPredictor` — In-Browser Linear Regression

```javascript
// Features used for in-browser prediction
features = [
    1,                        // bias term
    signalStrength / 100,
    latitude,
    longitude,
    networkTypeEncoded,       // 5G=1.0, 4G=0.8, LTE=0.6, 3G=0.4
    dataThroughput / 100,
    latency / 1000
]
// Solved via Gauss-Jordan elimination (matrix inverse)
// Target: signalQuality / 100
```

### Stack

| Component | Library | Version |
|-----------|---------|---------|
| UI Framework | React | 18.2 |
| Charts | Chart.js | 4.4.0 |
| Transpiler | Babel Standalone | 7.23.5 |
| Styling | Custom CSS (glassmorphism) | — |

---

## 🗺 Coverage Maps

Two coverage map types are generated:

### 1. Folium HeatMap + Markers (`coverage_map.html`)

```python
# HeatMap intensity = signal strength
heat_data = [[lat, lon, signal_strength] for each row]
HeatMap(heat_data, radius=15, blur=10).add_to(m)

# Color-coded locality pins
color = 'green' if signal_quality > 70 else 'orange' if > 40 else 'red'
```

**Popup content per marker:** Locality, Signal (dBm), Quality (%), Throughput (Mbps), Latency (ms), Network Type, BB60C, srsRAN, BladeRFxA9

### 2. Plotly Density Heatmap

```python
px.density_mapbox(
    df, lat="Latitude", lon="Longitude",
    z="Signal Strength (dBm)",
    mapbox_style="open-street-map",   # No API token required
    color_continuous_scale="Viridis"
)
```

---

## 📊 ML Models

### Random Forest (Scikit-learn)

| Target | R² Score | RMSE | Use |
|--------|----------|------|-----|
| `Data Throughput (Mbps)` | Reported at runtime | Mbps | Throughput prediction |
| `QoS_Score` | Reported at runtime | Score units | QoS prediction |

**Features:** Signal Strength, Signal Quality, Latitude, Longitude, Hour, Locality (encoded), Network Type (encoded), Avg Instrument Measurement

### LightGBM (GridSearchCV)

| Target | Evaluation | Notes |
|--------|-----------|-------|
| `Data Throughput (Mbps)` | R² + RMSE | 3-fold CV, 81 parameter combinations |

### Random Forest Classifier

**Target:** `signal_label` — 4-class signal quality (Excellent / Good / Fair / Poor)

**Labeling logic (RSRP-based):**

| Signal Strength | Label |
|-----------------|-------|
| ≥ −80 dBm | Excellent |
| −95 to −80 dBm | Good |
| −105 to −95 dBm | Fair |
| < −105 dBm | Poor |

---

## 💾 Output Files

| File | Description |
|------|-------------|
| `processed_network_data.csv` | Cleaned data with engineered features + QoS scores |
| `network_analysis_summary.txt` | Key stats summary (measurements, avg QoS, coverage areas) |
| `coverage_map.html` | Interactive Folium map with heatmap + locality markers |
| `signal_metrics_cleaned.csv` | Fully validated, median-imputed dataset |
| `signal_metrics_updated.csv` | Final cleaned file with fixed Signal Quality zeros |
| `network_report.pdf` | Gradio-generated PDF with stats table + visualizations |
| `ml_output/summary_report.txt` | ML pipeline output — label distribution + column map |
| `histogram.png / linegraph.png / heatmap.png / scatter.png` | Individual visualization exports |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm \
            folium plotly scipy gradio reportlab pillow
```

### Run the Python Analysis (Google Colab or Local)

```python
from wireless_mobile_comms_ import MobileNetworkAnalyzer

analyzer = MobileNetworkAnalyzer()
analyzer.run_complete_analysis(file_path="signal_metrics.csv")
```

Or run the complete script directly:

```bash
python wireless_mobile_comms_.py
```

### Launch the Gradio UI

The Gradio interface launches automatically at the end of the script:

```python
demo.launch(share=True)   # Generates public URL for Colab
```

### Run the React Dashboard

No build step needed:

```bash
# Simply open in any modern browser
open index.html
```
Or serve locally:
```bash
python -m http.server 8080
# then visit http://localhost:8080
```

Upload `signal_metrics.csv` or click **Load Sample Data** to explore immediately.

---

## 🛠 Tech Stack

| Tool | Role |
|------|------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat) | Core analytics engine |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat) | Data loading, cleaning, feature engineering |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat) | Random Forest, classification, preprocessing |
| ![LightGBM](https://img.shields.io/badge/-LightGBM-02569B?style=flat) | Gradient boosted throughput regression |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat) / ![Seaborn](https://img.shields.io/badge/-Seaborn-4C8CBF?style=flat) | 12-panel visualizations |
| ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white&style=flat) | Interactive density heatmap |
| ![Folium](https://img.shields.io/badge/-Folium-77B829?style=flat) | Leaflet.js-based coverage maps |
| ![Gradio](https://img.shields.io/badge/-Gradio-FF7C00?style=flat) | Web UI with PDF export |
| ![ReportLab](https://img.shields.io/badge/-ReportLab-CC0000?style=flat) | Automated PDF report generation |
| ![React](https://img.shields.io/badge/-React_18-61DAFB?logo=react&logoColor=black&style=flat) | Browser dashboard (CDN, no build) |
| ![Chart.js](https://img.shields.io/badge/-Chart.js-FF6384?logo=chartdotjs&logoColor=white&style=flat) | Real-time charts in the dashboard |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
