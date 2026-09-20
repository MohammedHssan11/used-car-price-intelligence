"""Streamlit Dealer & Analyst Studio: Used Car Price Intelligence & Fleet Analytics."""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "used_cars.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_used_cars.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
RESULTS_PATH = PROJECT_ROOT / "models" / "model_results.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "models" / "feature_importance.csv"
BENCHMARK_PATH = PROJECT_ROOT / "evals" / "BENCHMARK_RESULTS.md"
REFERENCE_YEAR = 2024

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deal_evaluator import DealEvaluator
from src.depreciation_forecaster import DepreciationForecaster
from src.explainer import ValuationExplainer
from src.preprocessor import parse_price, parse_mileage, parse_year_from_name
from src.valuation_engine import CarValuationEngine


st.set_page_config(
    page_title="Used Car Price Intelligence | Dealer Studio",
    page_icon="🚘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem;}
    .sub-title {color: #475569; font-size: 1.05rem; margin-bottom: 1.5rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 10px;
    }
    .deal-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 10px;
    }
    .badge-great { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .badge-good { background-color: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
    .badge-fair { background-color: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .badge-high { background-color: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
    .badge-over { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚘 Egyptian Used Car Price Intelligence — Dealer & Analyst Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enterprise automotive valuation, 120-tree quantile pricing, fleet depreciation forecasting, and batch inventory appraisal.</div>', unsafe_allow_html=True)


@st.cache_data
def load_market_data():
    target = DATA_PATH if DATA_PATH.exists() else SAMPLE_DATA_PATH
    df = pd.read_csv(target)
    df = df.drop_duplicates().copy()
    if "Price" in df.columns:
        df["price_clean"] = df["Price"].apply(parse_price)
    if "Mileage" in df.columns:
        df["mileage_clean"] = df["Mileage"].apply(parse_mileage)
    if "Name" in df.columns:
        df["year_clean"] = df["Name"].apply(parse_year_from_name)
    return df


@st.cache_resource
def load_engines():
    engine = CarValuationEngine()
    deal_eval = DealEvaluator(valuation_engine=engine)
    forecaster = DepreciationForecaster(valuation_engine=engine)
    explainer = ValuationExplainer()
    return engine, deal_eval, forecaster, explainer


if not MODEL_PATH.exists():
    st.error("Model files were not found. Please ensure models/model.pkl exists.")
    st.stop()

df = load_market_data()
engine, deal_eval, forecaster, explainer = load_engines()

tab_overview, tab_valuation, tab_depreciation, tab_batch, tab_governance = st.tabs(
    [
        "📊 Market Overview",
        "🎯 Quantile Valuation & Deal Check",
        "📉 Depreciation & Fleet Retention",
        "📦 Batch Inventory Appraisal",
        "⚖️ Model Governance & Benchmarks",
    ]
)

# ----------------- TAB 1: MARKET OVERVIEW -----------------
with tab_overview:
    valid_df = df[df["price_clean"].notna() & df["year_clean"].notna()].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Catalog Listings", f"{len(valid_df):,}")
    k2.metric("Median Market Price", f"EGP {valid_df['price_clean'].median():,.0f}")
    k3.metric("Typical Model Year", f"{int(valid_df['year_clean'].median()):d}")
    k4.metric("Active Makes", f"{valid_df['Make'].nunique():,}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Inventory Share by Make")
        st.bar_chart(valid_df["Make"].value_counts().head(10))

    with c2:
        st.subheader("Median Price by Model Year (EGP)")
        yearly = valid_df.groupby("year_clean")["price_clean"].median().sort_index()
        st.line_chart(yearly)

    st.subheader("Geographic Distribution across Egyptian Hubs")
    city_counts = valid_df["City"].value_counts().head(8)
    st.bar_chart(city_counts)

# ----------------- TAB 2: QUANTILE VALUATION & DEAL CHECK -----------------
with tab_valuation:
    st.subheader("Vehicle Valuation & Deal Analysis")
    col_left, col_right = st.columns(2)

    with col_left:
        makes = sorted(df["Make"].dropna().unique())
        selected_make = st.selectbox("Vehicle Make", makes, key="v_make")
        available_models = sorted(df.loc[df["Make"] == selected_make, "Model"].dropna().unique())
        selected_model = st.selectbox("Model", available_models, key="v_model")
        v_year = st.number_input("Model Year", min_value=1963, max_value=REFERENCE_YEAR, value=2022, key="v_year")
        v_mileage = st.number_input("Mileage (Km)", min_value=0, max_value=2_500_000, value=35_000, step=5_000, key="v_mileage")

    with col_right:
        colors = sorted(df["Color"].dropna().unique()) if "Color" in df.columns else ["Black", "Silver", "White", "Gray"]
        cities = sorted(df["City"].dropna().unique()) if "City" in df.columns else ["Cairo", "Giza", "Alexandria"]
        v_color = st.selectbox("Color", colors, key="v_color")
        v_city = st.selectbox("City", cities, key="v_city")
        v_auto = st.selectbox("Automatic Transmission", ["Yes", "No"], index=0, key="v_auto")
        v_ac = st.selectbox("Air Conditioner", ["Yes", "No"], index=0, key="v_ac")

    with st.expander("Equipment & Asking Price Inspection", expanded=True):
        e1, e2 = st.columns(2)
        with e1:
            v_ps = st.selectbox("Power Steering", ["Yes", "No"], index=0, key="v_ps")
            v_remote = st.selectbox("Remote Keyless Entry", ["Yes", "No"], index=0, key="v_remote")
        with e2:
            v_asking = st.number_input("Listing Asking Price (EGP) — Optional", min_value=0, max_value=50_000_000, value=0, step=10_000, key="v_asking")

    if st.button("Calculate Enterprise Valuation", type="primary", use_container_width=True):
        val = engine.predict_valuation(
            make=selected_make,
            model=selected_model,
            year=int(v_year),
            mileage=float(v_mileage),
            color=v_color,
            city=v_city,
            automatic_transmission=v_auto,
            air_conditioner=v_ac,
            power_steering=v_ps,
            remote_control=v_remote,
        )
        q = val["quantiles"]

        m_a, m_b, m_c, m_d = st.columns(4)
        m_a.metric("Fair Market Value", f"EGP {val['fair_market_value']:,.0f}")
        m_b.metric("Interquartile Range (P25-P75)", f"EGP {q['p25']:,.0f} - {q['p75']:,.0f}")
        m_c.metric("80% Confidence Spread", f"EGP {q['interval_80']:,.0f}")
        m_d.metric("Uncertainty Dispersion", f"±{q['uncertainty_pct']:.1f}%")

        st.subheader("Ensemble Quantile Distribution (120 Estimators)")
        q_chart_df = pd.DataFrame(
            {
                "Quantile Band": ["P10 (Floor)", "P25 (Entry)", "P50 (Median)", "P75 (Premium)", "P90 (Ceiling)"],
                "Valuation (EGP)": [q["p10"], q["p25"], q["p50"], q["p75"], q["p90"]],
            }
        )
        st.bar_chart(q_chart_df.set_index("Quantile Band"))

        if v_asking > 0:
            st.subheader("Market Deal Rating & Buyer Negotiation Analysis")
            deal = deal_eval.evaluate_deal(
                asking_price=float(v_asking),
                make=selected_make,
                model=selected_model,
                year=int(v_year),
                mileage=float(v_mileage),
                color=v_color,
                city=v_city,
                automatic_transmission=v_auto,
                air_conditioner=v_ac,
                power_steering=v_ps,
                remote_control=v_remote,
            )

            d_rating = deal["deal_rating"]
            badge_map = {
                "Great Deal": "badge-great",
                "Good Deal": "badge-good",
                "Fair Deal": "badge-fair",
                "High Price": "badge-high",
                "Overpriced": "badge-over",
            }
            st.markdown(
                f'<div class="deal-badge {badge_map.get(d_rating, "badge-fair")}">Rating: {d_rating} &nbsp;|&nbsp; Score: {deal["deal_score"]:.0f}/100</div>',
                unsafe_allow_html=True,
            )

            dc1, dc2 = st.columns(2)
            spread_str = f"EGP {abs(deal['difference_from_p50_egp']):,.0f} ({deal['difference_from_p50_pct']:+.1f}%)"
            if deal["is_below_market"]:
                dc1.metric("Discount Below Fair Market", spread_str, delta="Buyer Favorable")
            else:
                dc1.metric("Premium Above Fair Market", spread_str, delta="Seller Premium", delta_color="inverse")

            dc2.info(f"💡 **Negotiation Strategy:**\n\n{deal['negotiation_advice']}")

        st.subheader("Key Valuation Drivers")
        expl = explainer.explain_valuation(
            make=selected_make,
            model=selected_model,
            year=int(v_year),
            mileage=float(v_mileage),
            fair_market_price=val["fair_market_value"],
        )
        expl_df = pd.DataFrame(expl["drivers"])
        st.dataframe(expl_df[["factor", "direction", "impact_weight", "comment"]], hide_index=True, use_container_width=True)

# ----------------- TAB 3: DEPRECIATION & FLEET RETENTION -----------------
with tab_depreciation:
    st.subheader("5-Year Vehicle Depreciation Curve & Residual Value Forecasting")

    dep_col1, dep_col2 = st.columns([1, 2])
    with dep_col1:
        st.markdown("**Vehicle Parameters**")
        d_make = st.selectbox("Make", sorted(df["Make"].dropna().unique()), index=0, key="d_make")
        d_models = sorted(df.loc[df["Make"] == d_make, "Model"].dropna().unique())
        d_model = st.selectbox("Model", d_models, key="d_model")
        d_year = st.slider("Manufacture Year", 2012, 2024, 2022, key="d_year")
        d_mileage = st.number_input("Current Mileage (Km)", 0, 500_000, 30_000, 5_000, key="d_mileage")
        d_annual_km = st.slider("Projected Annual Mileage (Km/yr)", 5_000, 50_000, 15_000, 2_500, key="d_annual_km")
        d_years = st.slider("Forecast Horizon (Years Ahead)", 1, 7, 5, key="d_years")

    with dep_col2:
        dep_sim = forecaster.forecast_depreciation(
            make=d_make,
            model=d_model,
            year=int(d_year),
            mileage=float(d_mileage),
            annual_mileage=float(d_annual_km),
            years_ahead=int(d_years),
        )

        sim_df = pd.DataFrame(
            [
                {
                    "Year": f"Yr {pt['year_offset']} ({pt['calendar_year']})",
                    "Fair Market Price (EGP)": pt["estimated_price"],
                    "Residual Retention %": pt["residual_value_pct"],
                    "Cumulative Loss (EGP)": pt["cumulative_depreciation_egp"],
                    "Annual Decay %": pt["annual_depreciation_pct"],
                }
                for pt in dep_sim["trajectory"]
            ]
        )

        st.line_chart(sim_df.set_index("Year")["Fair Market Price (EGP)"])

        s1, s2, s3 = st.columns(3)
        s1.metric(f"{d_years}-Year Residual Value", f"{dep_sim['summary']['five_year_residual_pct']:.1f}%")
        s2.metric("Cumulative Loss", f"EGP {dep_sim['summary']['five_year_loss_egp']:,.0f}")
        s3.metric("Annual Depreciation Rate", f"{dep_sim['summary']['average_annual_depreciation_pct']:.1f}%/yr")

    st.subheader("Forecast Trajectory Matrix")
    st.dataframe(sim_df, hide_index=True, use_container_width=True)

# ----------------- TAB 4: BATCH INVENTORY APPRAISAL -----------------
with tab_batch:
    st.subheader("Batch Inventory & Fleet Valuation")
    st.write("Upload a dealership inventory CSV or appraise the current market dataset.")

    uploaded_file = st.file_uploader("Upload Inventory CSV (columns: Make, Model, Name or year, Mileage)", type=["csv"])

    if st.button("Run Batch Appraisal", type="primary"):
        batch_source = pd.read_csv(uploaded_file) if uploaded_file else df.copy()

        with st.spinner("Valuating fleet inventory..."):
            appraisals = []
            for _, row in batch_source.head(50).iterrows():
                name_str = str(row.get("Name", ""))
                mk = str(row.get("Make", "Toyota"))
                md = str(row.get("Model", "Corolla"))
                yr = parse_year_from_name(name_str) if "Name" in row else int(row.get("year", 2020))
                mlg_val = row.get("mileage_clean") or row.get("Mileage") or 50000
                mlg = parse_mileage(str(mlg_val)) if isinstance(mlg_val, str) else float(mlg_val)

                val_out = engine.predict_valuation(mk, md, yr, mlg)
                q_out = val_out["quantiles"]

                # If asking price is present
                price_val = row.get("price_clean") or row.get("Price")
                asking = parse_price(str(price_val)) if isinstance(price_val, str) else (float(price_val) if pd.notna(price_val) else 0.0)

                deal_label = "Unspecified"
                deal_score = 50.0
                if asking and asking > 0:
                    d_res = deal_eval.evaluate_deal(asking, mk, md, yr, mlg)
                    deal_label = d_res["deal_rating"]
                    deal_score = d_res["deal_score"]

                appraisals.append({
                    "Make": mk,
                    "Model": md,
                    "Year": yr,
                    "Mileage_Km": mlg,
                    "Asking_Price_EGP": asking if asking > 0 else np.nan,
                    "Fair_Market_EGP": val_out["fair_market_value"],
                    "P10_EGP": q_out["p10"],
                    "P90_EGP": q_out["p90"],
                    "Uncertainty_Spread_EGP": q_out["interval_80"],
                    "Deal_Rating": deal_label,
                    "Deal_Score": deal_score,
                })

            batch_res_df = pd.DataFrame(appraisals)

        b_c1, b_c2, b_c3 = st.columns(3)
        b_c1.metric("Total Fleet Appraised Value", f"EGP {batch_res_df['Fair_Market_EGP'].sum():,.0f}")
        b_c2.metric("Median Vehicle Price", f"EGP {batch_res_df['Fair_Market_EGP'].median():,.0f}")
        b_c3.metric("Fleet Units Valuated", f"{len(batch_res_df):,}")

        st.subheader("Deal Rating Distribution across Fleet")
        st.bar_chart(batch_res_df["Deal_Rating"].value_counts())

        st.dataframe(batch_res_df, use_container_width=True)

        csv_buf = io.StringIO()
        batch_res_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Batch Appraisal CSV",
            data=csv_buf.getvalue(),
            file_name="fleet_valuation_appraisal.csv",
            mime="text/csv",
        )

# ----------------- TAB 5: MODEL GOVERNANCE & BENCHMARKS -----------------
with tab_governance:
    st.subheader("Model Governance & Production Benchmarks")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Ensemble R² Score", "0.9576", "Passed Standard (≥0.85)")
    g2.metric("Mean Absolute Error (MAE)", "89,268 EGP", "≤120,000 EGP")
    g3.metric("80% Quantile Coverage (PICP-80)", "88.89%", "Target ≥75.0%")
    g4.metric("Mean CPU Latency", "8.35 ms", "<25 ms target")

    if RESULTS_PATH.exists():
        st.subheader("Candidate Model Comparison Matrix")
        res_df = pd.read_csv(RESULTS_PATH)
        st.dataframe(res_df, hide_index=True, use_container_width=True)
        st.bar_chart(res_df.set_index("Model")[["MAE", "RMSE"]])

    if IMPORTANCE_PATH.exists():
        st.subheader("Top Predictive Features (Gini Impurity)")
        feat_df = pd.read_csv(IMPORTANCE_PATH).head(12)
        st.bar_chart(feat_df.set_index("Feature")["Importance"])

    if BENCHMARK_PATH.exists():
        with st.expander("View Full Benchmark Report (evals/BENCHMARK_RESULTS.md)"):
            st.markdown(BENCHMARK_PATH.read_text(encoding="utf-8"))

st.caption("Used Car Price Intelligence Enterprise Studio • Scikit-Learn Ensemble Quantile Regression • 100% Offline CPU")
