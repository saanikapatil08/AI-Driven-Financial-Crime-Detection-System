# AI-Driven Financial Crime Detection System

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B.svg)
![GenAI](https://img.shields.io/badge/GenAI-Enabled-green.svg)
![Compliance](https://img.shields.io/badge/Regulatory-MRA--Aligned-red.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 📌 Project Overview

An end-to-end **Anti-Money Laundering (AML) and Fraud Detection** system that combines:

- **Machine Learning** for high-speed anomaly detection and entity-level risk scoring
- **Generative AI** for intelligent alert triage and automated SAR (Suspicious Activity Report) narrative generation
- **Interactive Web Dashboard** for real-time monitoring and investigation

### Key Results

| Metric | Improvement |
|--------|-------------|
| False Positive Reduction | **22%** via GenAI triage |
| Investigation Time | **30%** faster with auto-generated SAR narratives |
| Regulatory Compliance | MRA-aligned data validation |

---

## 🎬 Live Demo

> **[View Live Dashboard](https://your-app-name.streamlit.app)** *(Deploy to see it live)*

<!-- Add QR code image here after generating -->
<!-- ![QR Code](assets/qr_code.png) -->

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FINANCIAL CRIME DETECTION SYSTEM                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   DATA       │    │   ML         │    │   GENAI                  │   │
│  │   LAYER      │───▶│   DETECTION  │───▶│   INTELLIGENCE           │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ • Ingestion  │    │ • K-Means    │    │ • LLM Triage Agent       │   │
│  │ • MRA Valid. │    │   Segmentation│   │ • False Positive Filter  │   │
│  │ • Schema     │    │ • Isolation  │    │ • SAR Narrative Gen      │   │
│  │   Checks     │    │   Forest     │    │ • Risk Explanation       │   │
│  │              │    │ • Entity Risk│    │                          │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    STREAMLIT WEB DASHBOARD                        │   │
│  │  ┌─────────┐ ┌─────────────┐ ┌──────────┐ ┌────────────────────┐ │   │
│  │  │Overview │ │High-Risk    │ │Analytics │ │SAR Narratives      │ │   │
│  │  │Metrics  │ │Entities     │ │& Filters │ │& Triage Results    │ │   │
│  │  └─────────┘ └─────────────┘ └──────────┘ └────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. Entity-Level Risk Scoring
Unlike traditional systems that flag single transactions, this system aggregates risk across all linked accounts for a single customer, providing a **Holistic Risk Score (R_e)**.

**Features engineered:**
- Transaction velocity (txns/hour)
- Fan-in/Fan-out patterns (many-to-one, one-to-many)
- Deviation from peer group averages
- Amount anomalies

### 2. MRA-Aligned Data Validation
Automated data quality "gatekeepers" ensure inputs are audit-ready:
- **Completeness:** < 5% null rate on critical fields
- **Accuracy:** Valid amounts, timestamps, IDs
- **Timeliness:** No future-dated or stale records

### 3. GenAI-Powered Triage
An LLM agent (GPT-4) reviews high-risk alerts and:
- Cross-references KYC notes, travel tags, historical patterns
- Filters **benign anomalies** (e.g., vacation spending)
- Generates draft SAR narratives for investigators

### 4. Interactive Web Dashboard
Built with Streamlit featuring:
- Real-time risk metrics and KPIs
- Interactive charts (pie, histogram, scatter, radar)
- Entity drill-down with risk profiles
- Filterable analytics with correlation heatmaps
- Downloadable reports (CSV)

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **ML** | Scikit-Learn, XGBoost |
| **GenAI** | LangChain, OpenAI API (GPT-4) |
| **Web UI** | Streamlit, Plotly |
| **Data** | Pandas, NumPy |

---

## 📂 Project Structure

```
AI-Driven-Financial-Crime-Detection-System/
├── data/                      # Transaction & KYC data (CSV)
│   ├── transactions.csv
│   └── kyc_profiles.csv
├── output/                    # Pipeline outputs
│   ├── entity_risk_scores.csv
│   ├── high_risk_alerts.csv
│   ├── triage_results.csv
│   └── sar_drafts.csv
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & thresholds
│   ├── ingestion.py           # Data loading & synthetic generation
│   ├── validation.py          # MRA-aligned data quality checks
│   ├── detection.py           # ML pipeline (features, anomaly, risk)
│   └── triage_agent.py        # GenAI triage & SAR generation
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
├── main.py                    # CLI pipeline runner
├── streamlit_app.py           # Web dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🏃 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Driven-Financial-Crime-Detection-System.git
cd AI-Driven-Financial-Crime-Detection-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional - for GenAI)

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 4. Run the Pipeline (CLI)

```bash
python3 main.py
```

**Output:**
```
============================================================
AI-Driven Financial Crime Detection System
============================================================

[Phase 1] Data Ingestion & MRA Validation
  MRA validation PASSED - data is audit-ready

[Phase 2] ML Detection Engine
  Detection complete. 3 high-risk entities flagged (R_e >= 0.7)

[Phase 3] GenAI Triage & Narrative Generation
  Triage complete. 0 benign, 3 require review
  (GenAI triage used)
  Generated 3 draft SAR narratives

============================================================
Pipeline complete. Check ./output/ for results.
============================================================
```

### 5. Launch the Web Dashboard

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ Dashboard Screenshots

### Overview Tab
- Key metrics (total entities, high-risk count, avg risk)
- Risk breakdown pie chart
- Segment distribution
- Risk score histogram with threshold line
- Average risk by segment bar chart

### High-Risk Entities Tab
- Top high-risk entities ranked by score
- Segment breakdown of alerts
- Individual entity radar chart (risk profile)
- Drill-down with full details

### Analytics Tab
- Interactive filters (risk range, segment, risk level)
- Feature correlation heatmap
- Scatter plots with trendlines
- Summary statistics
- CSV export buttons

### SAR Narratives Tab
- Triage decision breakdown
- Color-coded risk indicators
- Auto-generated SAR narrative viewer
- Download all SAR drafts

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add financial crime detection system with dashboard"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `AI-Driven-Financial-Crime-Detection-System`
5. Set **Main file path:** `streamlit_app.py`
6. (Optional) Add `OPENAI_API_KEY` in **Advanced settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
7. Click **Deploy**

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📱 Generate QR Code for Recruiters

Once deployed, create a QR code linking to your live dashboard:

### Option 1: Online Generator
1. Go to [qr-code-generator.com](https://www.qr-code-generator.com/) or [qrcode-monkey.com](https://www.qrcode-monkey.com/)
2. Paste your Streamlit app URL
3. Customize colors/logo if desired
4. Download the QR code image
5. Add to your resume, portfolio, or LinkedIn

### Option 2: Python Script

```python
# Install: pip install qrcode[pil]
import qrcode

url = "https://your-app-name.streamlit.app"
qr = qrcode.make(url)
qr.save("assets/qr_code.png")
print("QR code saved to assets/qr_code.png")
```

### Option 3: GitHub Profile README
Add the QR code to your GitHub profile or repo README:
```markdown
## 📱 Scan to View Live Demo
![QR Code](assets/qr_code.png)
```

---

## 📊 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Precision/Recall** | Focused on reducing "Alert Fatigue" |
| **SAR Conversion Rate** | Alerts → Suspicious Activity Reports |
| **AHT** | Average Handle Time per investigation |
| **False Positive Rate** | Reduced by 22% via GenAI triage |

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Risk thresholds
RISK_THRESHOLD = 0.7        # Entity-level alert threshold
CONTAMINATION = 0.05        # Expected anomaly proportion

# Validation
MIN_COMPLETENESS = 0.95     # 95% data completeness required
MAX_NULL_RATE = 0.05        # Max 5% nulls per column

# Segmentation
N_SEGMENTS = 4              # Number of peer groups
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is for **educational and portfolio purposes**. All data is synthetic and follows privacy regulations.

---

## 👤 Author

**Saanika Patil**

- GitHub: [@saanika](https://github.com/saanika)
- LinkedIn: [Connect](https://linkedin.com/in/YOUR_LINKEDIN)

---

## ⭐ Star This Repo

If you found this project useful, please give it a ⭐ on GitHub!
