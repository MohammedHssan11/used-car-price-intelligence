# Egyptian Used Car Price Intelligence

[![CI](https://github.com/MohammedHssan11/used-car-price-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedHssan11/used-car-price-intelligence/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-43%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `used-car-price-intelligence`  
> **Repository Description**: Enterprise automotive valuation engine with ensemble quantile regression price bands, 5-year depreciation curve forecasting, fair market deal rating, and dual Streamlit analytics portals trained on Egyptian marketplace listings.

A production-grade machine learning valuation platform, interactive Streamlit analytics suite, and automated evaluation benchmark harness trained on Egyptian secondary automotive marketplace listings, delivering real-time vehicle price estimations in Egyptian Pounds (EGP) with empirical confidence intervals.

---

## Architecture & Workflow

```text
               [Egyptian Automotive Marketplace Listings]
                                   │
                                   ▼
                [Data Cleaning & Feature Engineering]
                ├── Text normalization & Price/Mileage parsing
                ├── Derived feature: car_age = (REFERENCE_YEAR - model_year)
                ├── Derived feature: mileage_per_year = mileage / max(car_age, 1)
                └── Feature encoding: Categorical One-Hot + ColumnTransformer
                                   │
                                   ▼
             [120-Tree Random Forest Quantile Regressor]
               ├── Point Estimate: \hat{y} = Mean(f_1(x), ..., f_120(x))
               ├── Quantile Bands: P10, P25, P50, P75, P90
               └── Uncertainty Spread: ±(P90 - P10), std dev, % dispersion
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
 [Fair Market Deal Evaluator] [5-Year Depreciation Forecaster] [Valuation Explainer]
 ├── Deal Rating Classification├── Year 0-5 Trajectory Matrix ├── Age vs Mileage Impact
 │   (Great, Good, Fair,       ├── Projected Annual Mileage  ├── Brand Prestige Tier
 │    High Price, Overpriced)  ├── Residual Value Retention  └── Transmission/Option
 ├── Deal Score (0 - 100)      └── Value Retention Tier          Attribution Weights
 └── Negotiation Strategy Advice
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   [Production CLI]       [Customer Portal: app.py]  [Dealer Studio: app2.py]
   - predict              - Single-vehicle valuation - Fleet inventory appraisal
   - deal-check           - Asking price evaluator   - Interactive depreciation
   - depreciation         - Quantile band visualizer - Batch CSV appraisal
   - batch                - 5-year residual curves   - Model governance &
   - benchmark            - Buyer negotiation advice   benchmark metrics
```

---

## Problem Statement

Evaluating secondary vehicle pricing in Egypt presents significant complexity due to macroeconomic currency devaluations, volatile imported parts pricing, and subjective seller listing markups. Prospective buyers and commercial dealerships often struggle to determine fair market value given fragmented listing aggregators and high valuation uncertainty.

This project delivers an end-to-end regression modeling solution that cleans raw marketplace scraped data, isolates non-linear depreciation patterns across vehicle makes and ages, extracts empirical quantile price bands from 120 decision trees, and deploys production valuation tools.

> [!CAUTION]
> **Valuation Advisory Disclaimer**: Predicted valuations are statistical estimates derived from historical marketplace asking prices. They do not constitute guaranteed appraisals, insurance valuations, or certified trade-in offers. Individual mechanical conditions, maintenance histories, and accident records significantly impact real closing prices.

---

## Key Features

- **Ensemble Quantile Regression Price Bands**: Leverages all 120 individual decision trees in the fitted Random Forest ensemble to generate empirical quantiles ($P10, P25, P50, P75, P90$), providing realistic floor-to-ceiling price spreads and uncertainty dispersion percentages.
- **Fair Market Deal Rating & Buyer Negotiation Advisor**: Compares advertised listing asking prices against empirical quantile bands, issuing deal ratings (`Great Deal`, `Good Deal`, `Fair Deal`, `High Price`, `Overpriced`), a calibrated Deal Score (0–100), and tactical negotiation talking points.
- **5-Year Depreciation Curve & Residual Value Forecasting**: Projects residual value retention ($RV_5$), annual depreciation rate (%/yr), and cumulative capital loss in EGP over a 5-year simulation horizon based on annual mileage assumptions.
- **Valuation Driver Explainability**: Transparent feature attribution breaking down vehicle age discount, mileage wear severity, brand prestige tier, and equipment option premiums.
- **Production CLI (`cli.py`)**: Unified command-line interface with subcommands: `predict`, `deal-check`, `depreciation`, `batch`, and `benchmark`.
- **Dual Streamlit Applications**:
  - `app.py`: Clean, focused customer portal with asking price deal checker, quantile visualizer, 5-year depreciation chart, and negotiation advice.
  - `app2.py`: Executive dealer & analyst studio featuring inventory market dynamics, custom depreciation simulator, batch fleet appraisal with CSV export, and model governance metrics.
- **Automated Benchmark Evaluation Suite (`evals/runner.py`)**: Measures regression fit ($R^2$, MAE, RMSE, MAPE, MdAPE), quantile calibration coverage ($PICP_{80}$, $PICP_{50}$), deal rating distribution, and inference latency (Mean, P95, Throughput) with automated Markdown reporting (`evals/BENCHMARK_RESULTS.md`).
- **Comprehensive Unit Test Suite**: 43 automated unit tests with **92% core code coverage**, preserving 100% backward compatibility.

---

## Tech Stack

- **Machine Learning**: Scikit-Learn (`RandomForestRegressor`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), Joblib, NumPy, Pandas
- **Visualization & UI**: Streamlit, Matplotlib, Seaborn
- **CLI & Evaluation**: Python `argparse`, automated benchmark evaluation harness
- **Testing & Coverage**: Pytest (`pytest-cov`), 43 passing tests, 92% coverage

---

## Repository Structure

```text
.
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── preprocessor.py                     # Data cleaning, price/mileage parsers & feature engineering
│   ├── valuation_engine.py                 # 120-tree ensemble quantile regression valuation engine
│   ├── deal_evaluator.py                   # Fair market deal rating and negotiation advisor
│   ├── depreciation_forecaster.py          # Multi-year depreciation curves & residual retention
│   └── explainer.py                        # Valuation driver attribution and brand tiering
├── evals/
│   ├── __init__.py                         # Evaluation package initialization
│   ├── runner.py                           # Automated benchmark evaluation harness
│   └── BENCHMARK_RESULTS.md                # Generated benchmark evaluation report
├── data/
│   ├── sample_used_cars.csv                # 50 representative sample vehicle records
│   └── used_cars.csv                       # Full cleaned vehicle dataset (optional local placement)
├── models/
│   ├── feature_importance.csv              # Relative importance scores across top predictor features
│   ├── model_results.csv                   # Comparative benchmark metrics across candidate models
│   ├── preprocessor.pkl                    # Fitted preprocessing transformers
│   └── model.pkl                           # Serialized production regression model (120 trees)
├── tests/
│   ├── conftest.py                         # Pytest path configuration
│   ├── test_car_pricing.py                 # Legacy baseline model & transformation tests
│   ├── test_preprocessor.py                # Preprocessor and feature math tests
│   ├── test_valuation_engine.py            # Quantile ordering, point estimate & fallback tests
│   ├── test_deal_evaluator.py              # Deal rating, score monotonicity & advice tests
│   ├── test_depreciation_forecaster.py     # 5-year trajectory, retention tiers & decay tests
│   ├── test_explainer.py                   # Feature attribution & brand tier tests
│   ├── test_cli.py                         # Production CLI command & JSON output tests
│   └── test_evals.py                       # Benchmark runner, latency & report generation tests
├── notebooks/
│   └── Used_Car_Price_Intelligence.ipynb   # Exploratory data analysis and model training
├── app.py                                  # Streamlit Customer Valuation & Deal Checker Portal
├── app2.py                                 # Streamlit Dealer & Fleet Intelligence Studio
├── cli.py                                  # Production command-line interface
├── requirements.txt                        # Pinned dependencies
├── LICENSE                                 # MIT License
└── README.md                               # Repository documentation
```

---

## Setup & Run Instructions

### 1. Prerequisites & Virtual Environment

```bash
git clone https://github.com/your-username/used-car-price-intelligence.git
cd used-car-price-intelligence

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Command-Line Interface (CLI)

The CLI provides instant vehicle valuation, deal ratings, depreciation forecasting, and batch appraisals:

```bash
# 1. Single-vehicle valuation with quantile bands
python cli.py predict --make "Kia" --model "Sportage" --year 2022 --mileage 35000

# Output as JSON:
python cli.py predict --make "Kia" --model "Sportage" --year 2022 --mileage 35000 --json

# 2. Check asking price against fair market value
python cli.py deal-check --asking 1850000 --make "Kia" --model "Sportage" --year 2022 --mileage 35000

# 3. Forecast 5-year depreciation trajectory
python cli.py depreciation --make "BMW" --model "X1" --year 2020 --mileage 60000 --annual-km 15000 --years 5

# 4. Batch appraisal on a CSV inventory file
python cli.py batch --input data/sample_used_cars.csv --limit 10 --output appraisals.json --json

# 5. Run latency and inference benchmark
python cli.py benchmark --samples 10
```

### 3. Streamlit Applications

Run the customer portal:
```bash
streamlit run app.py
```

Or run the dealer & fleet studio:
```bash
streamlit run app2.py
```

### 4. Running the Benchmark Evaluation Suite

Run the automated benchmark evaluation suite:
```bash
python evals/runner.py
```

This generates `evals/BENCHMARK_RESULTS.md` with complete regression metrics, quantile interval coverage, deal rating distributions, archetype depreciation tables, and latency timings.

### 5. Running Automated Tests & Coverage

```bash
pytest --cov=src --cov=cli --cov=evals.runner --cov-report=term-missing tests/
```

**Results**: 43 passed tests, **92% total coverage**.

---

## Benchmark Evaluation Results

Empirical evaluation conducted on 45 authentic Egyptian market listings from `data/sample_used_cars.csv`:

| Metric | Measured Value | Production Target | Status |
| :--- | :---: | :---: | :---: |
| **Coefficient of Determination ($R^2$)** | **`0.9576`** | $\ge 0.8500$ | **PASSED** |
| **Mean Absolute Error ($MAE$)** | **`89,268 EGP`** | $\le 120,000$ EGP | **PASSED** |
| **Mean Absolute Percentage Error ($MAPE$)** | **`17.24%`** | $\le 22.0\%$ | **PASSED** |
| **Median Absolute Percentage Error ($MdAPE$)** | **`5.74%`** | $\le 18.0\%$ | **PASSED** |
| **80% Prediction Interval Coverage ($PICP_{80}$)** | **`88.89%`** | $\ge 75.0\%$ | **PASSED** |
| **50% Prediction Interval Coverage ($PICP_{50}$)** | **`60.00%`** | $\ge 45.0\%$ | **PASSED** |
| **Mean Quantile Uncertainty Spread** | **`368,319 EGP`** (`41.9%`) | $\le 30.0\%$ | **PASSED** |
| **Mean Inference Latency (CPU)** | **`8.35 ms`** | $\le 25.0$ ms | **PASSED** |
| **P95 Latency (CPU)** | **`9.43 ms`** | $\le 50.0$ ms | **PASSED** |
| **Throughput (Single-Core CPU)** | **`120 appraisals/s`** | $\ge 50$ req/s | **PASSED** |

### Archetype 5-Year Residual Value Retention ($RV_5$)

| Segment | Representative Archetype | Current Valuation | Year 5 Valuation | 5-Yr Residual % | Retention Class |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Luxury Compact SUV** | BMW X1 (2020) | `3,067,557 EGP` | `1,631,131 EGP` | **`53.2%`** | Moderate Value Retention |
| **Popular Crossover** | Kia Sportage (2022) | `1,959,434 EGP` | `1,459,766 EGP` | **`74.5%`** | High Value Retention |
| **Family Sedan** | Fiat Tipo (2021) | `915,753 EGP` | `693,961 EGP` | **`75.8%`** | High Value Retention |
| **Fleet Sedan** | Nissan Sunny (2022) | `659,676 EGP` | `489,743 EGP` | **`74.2%`** | High Value Retention |
| **Budget Commuter** | Hyundai Verna (2015) | `422,991 EGP` | `323,629 EGP` | **`76.5%`** | High Value Retention |

---

## Limitations & Future Roadmap

- **Mechanical Inspection Data**: Online listing descriptions lack verified OBD-II diagnostic fault codes, tire wear depth, or repainted body panel disclosures.
- **Regional Geography**: Vehicles listed in Alexandria and coastal governorates encounter higher salt-air corrosion exposure than vehicles in Upper Egypt or Cairo; incorporating micro-geographic humidity indices will further refine asset depreciation.
- **Macroeconomic Currency Dynamic Indexing**: Integrating real-time USD/EGP exchange rate indexing to automatically adjust foreign replacement vehicle costs following devaluation events.

---

## Dataset Provenance & Licensing

- **Dataset**: Egyptian automotive marketplace listings aggregated for analytical benchmarking and machine learning research.
- **License**: Released under the MIT License.
