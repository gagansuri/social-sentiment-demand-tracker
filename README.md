# 📊 Social Sentiment Demand Tracker

> **Predict retail demand direction from social media signals, before it shows up in sales data.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-green.svg)](https://patents.google.com)

---

## What This Does

This repository demonstrates how real-time social media sentiment signals can be used to predict **demand direction** for retail products — in both directions:

- 📈 **Positive signals** (viral trends, rising search interest, positive reviews) → predict demand growth → trigger replenishment
- 📉 **Negative signals** (complaints, recall queries, safety concerns, backlash) → predict demand decline → suppress orders, recommend markdowns

Most retail forecasting systems only react to sales data after demand has already changed. This system detects the signal **12–48 hours before** it materialises in point-of-sale data.

---

## Core Concepts

### Sentiment Demand Direction Score (SDDS)
A signed scalar in **[-1, +1]** computed per product entity from social media content using Aspect-Based Sentiment Analysis (ABSA). Positive values indicate demand-generating sentiment; negative values indicate demand-suppressing sentiment.

### Demand Direction Modifier (DDM)
A rolling weighted aggregate of the SDDS stream, where recent values are exponentially weighted. The DDM is the operational signal fed into the demand forecasting pipeline:

```
DDM = Σ (SDDS_t × e^(-λ(T-t))) / Σ e^(-λ(T-t))
```

Where λ controls the decay rate and T is the current timestamp.

### Sentiment Velocity
The first temporal derivative of the rolling SDDS — detects **accelerating** sentiment deterioration before the DDM itself crosses a negative threshold.

---

## Repo Structure

```
social-sentiment-demand-tracker/
├── README.md
├── requirements.txt
├── src/
│   ├── data_generator.py      # Synthetic social media data generator
│   ├── sdds_engine.py         # SDDS computation (ABSA pipeline)
│   ├── ddm_engine.py          # DDM + sentiment velocity computation
│   ├── demand_signal.py       # Main signal processing pipeline
│   └── visualizer.py          # Plotly chart library
├── app/
│   └── streamlit_app.py       # Interactive Streamlit demo
├── scripts/
│   └── run_demo.py            # CLI demo script
└── notebooks/
    └── social_sentiment_demand_tracker.ipynb
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/gagansuri/social-sentiment-demand-tracker
cd social-sentiment-demand-tracker
pip install -r requirements.txt

# 2. Run the CLI demo
python scripts/run_demo.py

# 3. Launch the Streamlit app
streamlit run app/streamlit_app.py

# 4. Open the notebook
jupyter notebook notebooks/social_sentiment_demand_tracker.ipynb
```

No API keys required — all data is synthetically generated.

---

## Example Output

| Product | SDDS (latest) | DDM | Sentiment Velocity | Recommended Action |
|---|---|---|---|---|
| Stanley Tumbler | +0.82 | +0.74 | +0.12 | 🟢 Accelerate replenishment |
| Crocs Classic | +0.31 | +0.28 | -0.04 | 🟡 Monitor — trend plateauing |
| Ninja Blender | -0.61 | -0.54 | -0.18 | 🔴 Suppress orders · Flag markdown |
| Oura Ring | -0.79 | -0.71 | -0.22 | 🔴 Return-to-supplier alert |

---

## Connection to Patent

This repository demonstrates the social media signal processing components of a patent-pending retail AI system:

> *"System and Method for Multi-Source Demand Signal Fusion and Real-Time Store-Level Inventory Forecasting Using Social Media Trend Analysis"*
> USPTO Provisional Application — Patent Pending

The SDDS and DDM concepts implemented here form the sentiment analysis layer of a broader multi-modal demand forecasting engine that also incorporates weather signals, community event data, and search intent classification.

---

## Tech Stack

- **Sentiment Analysis:** VADER (NLTK) — fast, lexicon-based, no GPU required
- **Data Processing:** pandas, numpy
- **Visualisation:** Plotly (interactive charts)
- **App:** Streamlit
- **Notebook:** Jupyter

---

## Author

**Gagan Suri** — AI in Retail | Patent Pending Inventor | [LinkedIn](https://linkedin.com/in/gagansuri)

*Follow my [Think Retail AI newsletter on LinkedIn](Subscribe on LinkedIn https://www.linkedin.com/build-relation/newsletter-follow?entityUrn=7468788953366818817) for weekly insights on how AI is transforming retail operations.*
