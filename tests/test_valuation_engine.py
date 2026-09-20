"""Unit tests for the automotive valuation engine and ensemble quantile regression."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessor import create_vehicle_dataframe
from src.valuation_engine import CarValuationEngine


@pytest.fixture(scope="module")
def valuation_engine() -> CarValuationEngine:
    return CarValuationEngine()


def test_valuation_engine_initialization(valuation_engine: CarValuationEngine) -> None:
    """Test successful initialization and pipeline steps."""
    assert valuation_engine.pipeline is not None
    assert valuation_engine.preprocessor is not None
    assert valuation_engine.regressor is not None


def test_predict_point_estimate(valuation_engine: CarValuationEngine) -> None:
    """Test mean price prediction on a standard vehicle."""
    df = create_vehicle_dataframe(
        make="Hyundai",
        model="Tucson Turbo GDI",
        year=2021,
        mileage=80_000,
    )
    price = valuation_engine.predict_point_estimate(df)
    assert price > 100_000.0
    assert isinstance(price, float)


def test_predict_quantiles_ordering(valuation_engine: CarValuationEngine) -> None:
    """Test monotonic ordering of quantile price bands P10 <= P25 <= P50 <= P75 <= P90."""
    df = create_vehicle_dataframe(
        make="Kia",
        model="Sportage",
        year=2022,
        mileage=40_000,
    )
    q = valuation_engine.predict_quantiles(df)

    assert q["p10"] > 0
    assert q["p10"] <= q["p25"]
    assert q["p25"] <= q["p50"]
    assert q["p50"] <= q["p75"]
    assert q["p75"] <= q["p90"]
    assert q["interval_80"] == pytest.approx(q["p90"] - q["p10"], 1e-2)
    assert q["std"] >= 0.0
    assert q["uncertainty_pct"] >= 0.0


def test_predict_valuation_wrapper(valuation_engine: CarValuationEngine) -> None:
    """Test end-to-end predict_valuation convenience function."""
    result = valuation_engine.predict_valuation(
        make="Toyota",
        model="Corolla",
        year=2019,
        mileage=95_000,
        city="Cairo",
    )
    assert "fair_market_value" in result
    assert "quantiles" in result
    assert "vehicle" in result
    assert result["vehicle"]["make"] == "Toyota"
    assert result["fair_market_value"] > 0.0


def test_valuation_fallback_empty_pipeline() -> None:
    """Test engine handling when pipeline is None."""
    dummy_engine = CarValuationEngine(pipeline=None)
    dummy_df = pd.DataFrame()
    assert dummy_engine.predict_point_estimate(dummy_df) == 0.0
    q = dummy_engine.predict_quantiles(dummy_df)
    assert q["mean"] == 0.0
    assert q["p50"] == 0.0


def test_valuation_non_ensemble_fallback() -> None:
    """Test fallback when regressor lacks estimators_."""
    class DummyPipeline:
        def predict(self, df):
            return [500000.0]

    dummy = CarValuationEngine(pipeline=DummyPipeline())
    df = create_vehicle_dataframe("Kia", "Cerato", 2018, 90000)
    q = dummy.predict_quantiles(df)
    assert q["mean"] == 500000.0
    assert q["p50"] == 500000.0
    assert q["p10"] < q["p50"] < q["p90"]
    assert q["uncertainty_pct"] > 0.0


def test_valuation_engine_missing_path() -> None:
    """Test initializing with non-existent path."""
    eng = CarValuationEngine(model_path="nonexistent_path/fake_model.pkl")
    assert eng.pipeline is None
    assert eng.preprocessor is None
    assert eng.regressor is None

