# WasteWise AI

**ML-based municipal solid waste generation estimation and collection-priority support platform**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-119%20passed-brightgreen)](#testing)

---

## Overview

Municipal authorities need useful estimates of waste generation and a straightforward way to identify cities or records that may require greater collection attention.

WasteWise AI addresses this by:

1. Learning from historical municipal solid waste (MSW) observations across Indian metropolitan and state-capital cities.
2. Training a supervised machine-learning regression model to estimate waste generation in tonnes per day.
3. Assigning a collection-priority band (LOW / MEDIUM / HIGH) derived from the estimated waste value.
4. Presenting the results through an interactive Streamlit dashboard.

**The system does not use real-time data, external APIs, or live sensor feeds.**  
All estimates are based on historical patterns in the source dataset.

---

## Key Features

| Feature | Description |
|---|---|
| Waste generation estimation | Predicts daily MSW generation (tonnes/day) for any city and year |
| Collection-priority classification | Assigns LOW / MEDIUM / HIGH based on predicted waste and training-data thresholds |
| Interactive Streamlit dashboard | Multi-page web application with light/dark theme |
| City analysis | Historical trend, observed vs predicted, per-capita waste |
| Historical analytics | Year-over-year trends, city comparison, top/bottom-city ranking |
| Model comparison | Side-by-side metrics for all evaluated models |
| Actual vs predicted analysis | Scatter plot and table on the chronological test set |
| Feature importance | Bar chart and table of Gradient Boosting feature importances |
| Model information page | Methodology, hyperparameters, evaluation results, limitations |
| Light / dark theme | Toggle in the sidebar on every page |
| Extrapolation warning | Visible alert when the selected year exceeds the latest observed year (2022) |
| Prediction-driver explanation | Plain-language breakdown of the three main model drivers per prediction |
| Chronological model evaluation | Train/test split respects temporal order — no random splitting |

---

## Dataset

**Source:** Ministry of Statistics and Programme Implementation (MoSPI), Government of India  
**Publication:** *EnviStats India 2024*, Volume II: Environment Accounts  
**Statement:** 3.06 — Municipal Solid Waste Generation in Metro Cities and State Capitals

| Property | Value |
|---|---|
| Cities in source table | 46 |
| Fiscal-year snapshots | 9 |
| Rows in extracted CSV | 414 |
| Modeling cities (after preprocessing) | 45 |
| Observed waste-generation records | 307 |

**Fiscal years present in the dataset:**

```
1999-00  |  2004-05  |  2010-11  |  2015-16  |  2018-19
2019-20  |  2020-21  |  2021-22  |  2022-23
```

Aurangabad was removed during preprocessing because its waste-generation values were unavailable across all available fiscal years.

> See [`docs/dataset.md`](docs/dataset.md) for full dataset documentation including the missing-data pattern, column descriptions, and data-quality notes.

---

## Problem Statement

Indian cities generate millions of tonnes of municipal solid waste daily, and the volume continues to grow with urbanisation. Efficient allocation of waste-collection resources requires an understanding of which cities generate the most waste and how volumes are changing over time. Official data collection is periodic rather than continuous, and many cities lack complete historical records.

This project builds a data-driven estimation tool that uses available historical records to support waste-management planning by:
- providing baseline waste-generation estimates for cities and fiscal years,
- classifying cities into relative priority bands,
- surfacing observed vs estimated differences for anomaly detection.

---

## Objectives

1. Analyse historical municipal solid waste generation across Indian metropolitan and state-capital cities.
2. Build an explainable supervised ML regression model for waste-generation estimation.
3. Evaluate the model using a chronological holdout that respects temporal order.
4. Convert estimated waste into operationally understandable LOW / MEDIUM / HIGH priority bands.
5. Provide an interactive dashboard for exploration, prediction, and analytics.
6. Present model performance and limitations transparently.

---

## Methodology

### Pipeline

```
Raw government data (EnviStats India 2024, Statement 3.06)
  → Data cleaning (drop constant/null columns, remove Aurangabad)
  → Feature engineering (year_start, log_population, pop_tier, log target)
  → Chronological train/test split (train: year_start < 2020)
  → Preprocessing inside sklearn Pipeline (StandardScaler + OneHotEncoder)
  → Candidate regression models trained on log-transformed target
  → Time-aware cross-validation (TimeSeriesSplit, 5 expanding-window folds)
  → Model selection (lowest test RMSE in original units)
  → Final evaluation on chronological holdout (2020-21 → 2022-23)
  → Prediction: expm1(log_prediction) → tonnes/day
  → Priority assignment (frozen training-set quantile thresholds)
  → Dashboard and analytics
```

### Modeling Features

| Feature | Type | Description |
|---|---|---|
| `year_start` | Numeric | Starting calendar year of the fiscal year |
| `log_population` | Numeric | Natural log of 2011 Census population |
| `pop_tier` | Numeric | 1 = Mega (>4 M), 2 = Large (2–4 M), 3 = Medium (<2 M) |
| `city` | Categorical | One-hot encoded; fitted on training data only |

### Excluded Variables (leakage prevention)

The following variables are present in the cleaned dataset for analytics but are **never used as model inputs**:

| Variable | Reason |
|---|---|
| `waste_generated` | The raw target |
| `log_waste_generated` | The transformed target |
| `waste_per_capita` | Computed directly from the target |
| `city_mean_waste_all` | Cross-year mean of the target — temporal leakage |

> See [`docs/methodology.md`](docs/methodology.md) for the complete technical pipeline description.

---

## Train / Test Strategy

A **chronological holdout** is used. Random splitting would allow future fiscal-year information to appear in the training set, inflating apparent performance.

| Set | Condition | Fiscal years | Observed rows |
|---|---|---|---|
| Training | `year_start < 2020` | 1999-00 → 2019-20 | **172** |
| Test | `year_start >= 2020` | 2020-21 → 2022-23 | **135** |

Rows with null `waste_generated` are excluded from both sets.

---

## Models Evaluated

All models trained on log-transformed target; metrics back-transformed to original units for reporting.

| Model | MAE (t/day) | RMSE (t/day) | R² | MAPE (%) |
|---|---|---|---|---|
| Dummy (mean) — baseline | 1,019.8 | 2,203.8 | −0.143 | 46.7 |
| Ridge Regression | 579.1 | 1,062.5 | 0.734 | 37.7 |
| Random Forest Regressor | 378.1 | 637.5 | 0.904 | 28.8 |
| **Gradient Boosting Regressor** | **405.9** | **581.7** | **0.920** | 34.8 |

Time-aware cross-validation (log-RMSE, 5 folds):

| Model | CV log-RMSE (mean) |
|---|---|
| Dummy | 1.040 |
| Ridge | 0.476 |
| Random Forest | 0.504 |
| **Gradient Boosting** | **0.463** |

---

## Selected Model: Gradient Boosting Regressor

**Algorithm:** `sklearn.ensemble.GradientBoostingRegressor`

**Hyperparameters:**

```python
n_estimators   = 300
learning_rate  = 0.05
max_depth      = 4
subsample      = 0.8
min_samples_leaf = 2
random_state   = 42
```

**Selection rationale:**
- Lowest test RMSE (581.7 t/day) on the chronological holdout
- Highest test R² (0.920)
- Best time-aware cross-validation log-RMSE (0.463)

> **R² = 0.920 is not "92 % accuracy."** R² measures the proportion of variance in waste generation explained by the model on the test set.

---

## Feature Importance

From the trained Gradient Boosting model (top features):

| Feature | Importance |
|---|---|
| `log_population` | 0.6949 |
| `year_start` | 0.1566 |
| `city_Dhanbad` | 0.0296 |
| `city_Mumbai` | 0.0142 |
| `city_Ranchi` | 0.0139 |
| `pop_tier` | 0.0068 |

Population scale and temporal trend together account for approximately 85 % of model importance. City-specific OHE dummies contribute the remaining ~15 % collectively.

> **Feature importance indicates model reliance, not causation.** A high importance score means the model found the feature useful for splitting decisions — it does not mean the feature causes waste generation.

---

## Priority Logic

Collection priority is **not** a separately trained classifier. It is a rule applied to the regression prediction after the fact.

| Priority | Threshold | Basis |
|---|---|---|
| 🟢 LOW | predicted waste ≤ 510.43 t/day | p33 of training-period observations |
| 🟡 MEDIUM | 510.43 < predicted waste ≤ 1,010.0 t/day | p33–p67 of training-period observations |
| 🔴 HIGH | predicted waste > 1,010.0 t/day | above p67 of training-period observations |

Thresholds are derived from the **training-period observed waste distribution only** (n = 172 rows, `year_start < 2020`). The test set is excluded from threshold construction to prevent information leakage.

---

## Application Pages

| Page | Description |
|---|---|
| **Home** | Landing page with navigation overview |
| **Dashboard** | KPIs, yearly waste trend, priority distribution pie chart, top/bottom city tables |
| **Prediction** | City/population/year inputs → estimated waste, priority, model drivers, historical context chart |
| **City Analysis** | Per-city historical trend, observed vs predicted table and chart, per-capita waste |
| **Analytics** | Yearly trend, city comparison, model comparison, actual vs predicted scatter, feature importance |
| **Model** | Model details, evaluation metrics, train/test strategy, priority thresholds, known limitations |

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### Steps

```bash
git clone <repository-url>
cd WasteWise-AI

# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify the model artifact is present

```bash
# The trained model must exist at this path for the application to run
ls models/best_model.joblib      # macOS / Linux
dir models\best_model.joblib     # Windows
```

If the model file is missing, reproduce it from the training script (see [Reproducibility](#reproducibility)).

---

## Usage

1. Start the application:

   ```bash
   streamlit run app.py
   ```

2. Open the URL shown in the terminal (default: `http://localhost:8501`).

3. Navigate to **Prediction** in the sidebar.

4. Select a city from the dropdown, or choose **— Custom city —** and enter a city name manually.

5. Verify or adjust the **2011 Census Population** field.  
   For dataset cities, the value is pre-filled from the source data.

6. Select a **Year** (fiscal year start).

7. Click **🔮 Estimate**.

8. Interpret the result:
   - **Estimated Waste** — predicted MSW generation in tonnes/day
   - **Collection Priority** — LOW / MEDIUM / HIGH based on training-set quantile thresholds
   - **Main Model Drivers** — plain-language breakdown of the top three contributing factors
   - If the year exceeds 2022, an **Extrapolation** warning is shown indicating the result should be validated before operational use

---

## Project Limitations

1. **Relatively small dataset** — 307 observed target rows across 45 cities and 9 fiscal years. Models may not generalise well to cities or time periods not well-represented in the data.

2. **Non-uniform time spacing** — fiscal-year snapshots are not evenly spaced (5-year, 6-year, and then annual gaps). The model uses `year_start` as a numeric covariate, not a proper time-series structure.

3. **Static 2011 population** — the population anchor is the 2011 Census value. Actual city populations have grown since then. Per-capita estimates are approximations.

4. **Missing waste observations** — 98 city-year rows have no recorded waste value. These are excluded from model training and evaluation and cannot be evaluated.

5. **Limited external explanatory variables** — no GDP, urbanisation rate, waste-treatment efficiency, climate, or infrastructure data is included (none available in the source dataset).

6. **Unseen-city / generalisation limitations** — for cities not present in the training set, the model has no city-specific learned pattern and relies solely on population and year trend.

7. **Tier 2 / Tier 3 performance** — on the test set, the model achieves negative R² for Tier 2 (2–4 M population) and Tier 3 (<2 M population) cities. The model is most reliable for Tier 1 mega-cities.

8. **Extrapolation beyond observed years** — predictions for years after 2022-23 extrapolate the learned time trend. Uncertainty increases with distance from the training period.

9. **Priority thresholds are dataset-relative** — thresholds are percentiles of this specific dataset. They are not transferable to a national or international waste-management standard without recalibration.

10. **Model estimates are not official data** — predictions should not be used for operational waste-management decisions without independent validation.

---

## Testing

The test suite uses [pytest](https://docs.pytest.org/).

```bash
python -m pytest tests/ -v
```

**Current result: 119 passed**

The suite covers:
- `tests/test_prediction.py` — unit tests for the prediction module (41 tests)
- `tests/test_priority.py` — unit tests for the priority module (44 tests)
- `tests/test_integration.py` — integration tests for the end-to-end pipeline (34 tests)

---

## Reproducibility

### Application (no retraining needed)

The trained model artifact (`models/best_model.joblib`) is included in the repository.  
Install dependencies and run `streamlit run app.py` as described in [Installation](#installation).

### Full pipeline reproduction

If you wish to regenerate all artefacts from the raw data:

```bash
# Phase 2: Clean data and engineer features
python notebooks/phase2_cleaning_features.py

# Phase 3: Train and evaluate models (overwrites best_model.joblib)
python notebooks/phase3_model_training.py

# Phase 4: Priority analysis
python notebooks/phase4_priority_analysis.py
```

> **Note:** Phase 2 requires `data/raw/india_msw_city_year.csv` (included).  
> Phase 3 overwrites `models/best_model.joblib` and `reports/phase3_*.csv`.

### Inspection notebooks

```bash
# Phase 1: Raw dataset inspection (read-only, modifies nothing)
python notebooks/phase1_dataset_inspection.py
```

---

## Environment Variables

This project does **not** use environment variables.  
No `.env` file is required.

---

## Project Structure

```
WasteWise-AI/
│
├── app.py                          # Streamlit application entry point (Home page)
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml                 # Theme and server configuration
│
├── pages/                          # Streamlit multi-page app pages
│   ├── 1_Dashboard.py
│   ├── 2_Prediction.py
│   ├── 3_City_Analysis.py
│   ├── 4_Analytics.py
│   └── 5_Model.py
│
├── src/                            # Core Python modules
│   ├── __init__.py
│   ├── prediction.py               # predict_waste(), explain_prediction()
│   ├── priority.py                 # Frozen thresholds, classify(), StrategyA/B
│   ├── analytics.py                # Historical data query functions
│   └── ui_utils.py                 # Shared Streamlit UI helpers and theme CSS
│
├── data/
│   ├── raw/
│   │   └── india_msw_city_year.csv # Original extracted dataset (never modified)
│   └── processed/
│       └── india_msw_cleaned.csv   # Phase 2 output (405 rows × 13 columns)
│
├── models/
│   └── best_model.joblib           # Trained Gradient Boosting pipeline
│
├── notebooks/                      # Phase scripts (run sequentially)
│   ├── phase1_dataset_inspection.py
│   ├── phase2_cleaning_features.py
│   ├── phase3_model_training.py
│   └── phase4_priority_analysis.py
│
├── tests/
│   ├── __init__.py
│   ├── test_prediction.py
│   ├── test_priority.py
│   └── test_integration.py
│
├── reports/
│   ├── phase3_model_results.csv
│   ├── phase3_feature_importance.csv
│   ├── phase4_priority_analysis.csv
│   └── figures/                    # Model evaluation figures (PNG)
│
└── docs/
    ├── dataset.md                  # Dataset documentation
    └── methodology.md              # Technical methodology
```

---

## Disclaimer

> Predictions produced by WasteWise AI are model estimates derived from historical municipal solid waste data.  
> They are **not** official waste-management data.  
> Do not use them for operational decisions without independent validation from authoritative sources.

---

## Acknowledgements

Dataset: Ministry of Statistics and Programme Implementation (MoSPI), Government of India — *EnviStats India 2024*, Volume II: Environment Accounts, Statement 3.06.
