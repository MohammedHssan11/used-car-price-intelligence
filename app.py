"""Streamlit Customer Portal: Used Car Price Intelligence & Fair Market Deal Evaluator."""

from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "used_cars.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_used_cars.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
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
    page_title="Used Car Price Intelligence",
    page_icon="🚗",
    layout="centered",
)


@st.cache_data
def load_choices():
    path = DATA_PATH if DATA_PATH.exists() else SAMPLE_DATA_PATH
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_engines():
    engine = CarValuationEngine()
    deal_eval = DealEvaluator(valuation_engine=engine)
    forecaster = DepreciationForecaster(valuation_engine=engine)
    explainer = ValuationExplainer()
    return engine, deal_eval, forecaster, explainer


st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.2rem; }
    .sub-header { color: #475569; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .deal-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
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

st.markdown('<div class="main-header">🚗 Egyptian Used Car Price Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered automotive valuation, 120-tree quantile uncertainty bands, and buyer negotiation advisory.</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model file (models/model.pkl) was not found. Please verify the pre-trained weights.")
    st.stop()

df = load_choices()
engine, deal_eval, forecaster, explainer = load_engines()

makes = sorted(df["Make"].dropna().unique())
default_make_idx = makes.index("Kia") if "Kia" in makes else 0

st.subheader("1. Vehicle Specifications")
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Make", makes, index=default_make_idx)
    available_models = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique())
    default_model_idx = available_models.index("Sportage") if "Sportage" in available_models else 0
    car_model = st.selectbox("Model", available_models, index=default_model_idx)
    year = st.number_input("Model Year", min_value=1963, max_value=REFERENCE_YEAR, value=2022)
    mileage = st.number_input("Mileage (Km)", min_value=0, max_value=2_500_000, value=35_000, step=5_000)

with col2:
    colors = sorted(df["Color"].dropna().unique()) if "Color" in df.columns else ["Black", "Silver", "White", "Gray"]
    cities = sorted(df["City"].dropna().unique()) if "City" in df.columns else ["Cairo", "Giza", "Alexandria"]
    color = st.selectbox("Exterior Color", colors, index=0)
    city = st.selectbox("Listing City / Location", cities, index=0)
    automatic = st.selectbox("Automatic Transmission", ["Yes", "No"], index=0)
    air_conditioner = st.selectbox("Air Conditioner", ["Yes", "No"], index=0)

with st.expander("Additional Options & Asking Price Check", expanded=True):
    opt1, opt2 = st.columns(2)
    with opt1:
        power_steering = st.selectbox("Power Steering", ["Yes", "No"], index=0)
        remote_control = st.selectbox("Remote Control / Keyless", ["Yes", "No"], index=0)
    with opt2:
        asking_price_input = st.number_input(
            "Listing Asking Price (EGP) — Optional",
            min_value=0,
            max_value=50_000_000,
            value=0,
            step=10_000,
            help="Enter the seller's asking price to evaluate whether it's a Great Deal, Fair Deal, or Overpriced.",
        )

if st.button("Evaluate Valuation & Market Deal", type="primary", use_container_width=True):
    with st.spinner("Executing 120-tree ensemble quantile regression..."):
        val_res = engine.predict_valuation(
            make=make,
            model=car_model,
            year=int(year),
            mileage=float(mileage),
            color=color,
            city=city,
            automatic_transmission=automatic,
            air_conditioner=air_conditioner,
            power_steering=power_steering,
            remote_control=remote_control,
        )

        quantiles = val_res["quantiles"]
        fair_price = val_res["fair_market_value"]

    st.markdown("---")
    st.subheader("2. Fair Market Valuation & Quantile Bands")

    m1, m2, m3 = st.columns(3)
    m1.metric("Fair Market Value", f"EGP {fair_price:,.0f}")
    m2.metric("Competitive Range (P25 - P75)", f"EGP {quantiles['p25']:,.0f} - {quantiles['p75']:,.0f}")
    m3.metric("Uncertainty Spread (±)", f"EGP {quantiles['interval_80']:,.0f}", f"{quantiles['uncertainty_pct']:.1f}% std")

    # Quantile Band Display
    q_data = pd.DataFrame(
        {
            "Percentile": ["P10 (Bargain)", "P25 (Competitive)", "P50 (Median)", "P75 (Premium)", "P90 (Ceiling)"],
            "Price (EGP)": [
                quantiles["p10"],
                quantiles["p25"],
                quantiles["p50"],
                quantiles["p75"],
                quantiles["p90"],
            ],
        }
    )
    st.bar_chart(q_data.set_index("Percentile"))

    # Asking Price Deal Checker
    if asking_price_input > 0:
        st.markdown("---")
        st.subheader("3. Asking Price Deal Analysis")

        deal = deal_eval.evaluate_deal(
            asking_price=float(asking_price_input),
            make=make,
            model=car_model,
            year=int(year),
            mileage=float(mileage),
            color=color,
            city=city,
            automatic_transmission=automatic,
            air_conditioner=air_conditioner,
            power_steering=power_steering,
            remote_control=remote_control,
        )

        rating = deal["deal_rating"]
        badge_cls = {
            "Great Deal": "badge-great",
            "Good Deal": "badge-good",
            "Fair Deal": "badge-fair",
            "High Price": "badge-high",
            "Overpriced": "badge-over",
        }.get(rating, "badge-fair")

        st.markdown(
            f'<div class="deal-badge {badge_cls}">Deal Rating: {rating} &nbsp;|&nbsp; Deal Score: {deal["deal_score"]:.0f}/100</div>',
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(2)
        diff_str = f"EGP {abs(deal['difference_from_p50_egp']):,.0f} ({deal['difference_from_p50_pct']:+.1f}%)"
        if deal["is_below_market"]:
            d1.metric("Buyer Savings", diff_str, delta="Below Fair Market", delta_color="normal")
        else:
            d1.metric("Seller Premium", diff_str, delta="Above Fair Market", delta_color="inverse")

        d2.info(f"💡 **Negotiation Strategy:**\n\n{deal['negotiation_advice']}")

    # 5-Year Depreciation Trajectory
    st.markdown("---")
    st.subheader("4. 5-Year Depreciation & Residual Value Forecast")

    dep = forecaster.forecast_depreciation(
        make=make,
        model=car_model,
        year=int(year),
        mileage=float(mileage),
        annual_mileage=15000,
        years_ahead=5,
        color=color,
        city=city,
    )

    dep_df = pd.DataFrame(
        [
            {
                "Year": f"Year {pt['year_offset']} ({pt['calendar_year']})",
                "Projected Valuation (EGP)": pt["estimated_price"],
                "Residual %": pt["residual_value_pct"],
            }
            for pt in dep["trajectory"]
        ]
    )

    st.line_chart(dep_df.set_index("Year")["Projected Valuation (EGP)"])

    summary = dep["summary"]
    r1, r2, r3 = st.columns(3)
    r1.metric("5-Year Residual Value", f"{summary['five_year_residual_pct']:.1f}%")
    r2.metric("5-Year Cumulative Loss", f"EGP {summary['five_year_loss_egp']:,.0f}")
    r3.metric("Retention Class", summary["retention_tier"].split(" (")[0])

    # Explainability Drivers
    st.markdown("---")
    st.subheader("5. Valuation Drivers & Feature Attribution")
    exp = explainer.explain_valuation(
        make=make,
        model=car_model,
        year=int(year),
        mileage=float(mileage),
        fair_market_price=fair_price,
    )

    for driver in exp["drivers"]:
        st.write(f"• **{driver['factor']}**: {driver['direction']} ({driver['impact_weight']}) — *{driver['comment']}*")

st.caption("Egyptian Used Car Price Intelligence • Powered by Scikit-Learn Random Forest Quantiles • 100% Offline CPU")
