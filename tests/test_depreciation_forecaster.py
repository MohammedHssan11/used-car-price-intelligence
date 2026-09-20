"""Unit tests for automotive multi-year depreciation forecasting."""

from __future__ import annotations

import pytest
from src.depreciation_forecaster import DepreciationForecaster
from src.valuation_engine import CarValuationEngine


@pytest.fixture(scope="module")
def forecaster() -> DepreciationForecaster:
    engine = CarValuationEngine()
    return DepreciationForecaster(valuation_engine=engine)


def test_forecast_depreciation_trajectory_length(forecaster: DepreciationForecaster) -> None:
    """Test trajectory contains Year 0 baseline plus 5 future forecast points."""
    result = forecaster.forecast_depreciation(
        make="Hyundai",
        model="Tucson Turbo GDI",
        year=2021,
        mileage=70_000,
        years_ahead=5,
    )
    trajectory = result["trajectory"]
    assert len(trajectory) == 6
    assert trajectory[0]["year_offset"] == 0
    assert trajectory[-1]["year_offset"] == 5


def test_forecast_depreciation_residuals(forecaster: DepreciationForecaster) -> None:
    """Test residual percentages decrease over time."""
    result = forecaster.forecast_depreciation(
        make="Toyota",
        model="Corolla",
        year=2020,
        mileage=60_000,
        years_ahead=5,
    )
    traj = result["trajectory"]
    assert traj[0]["residual_value_pct"] == 100.0
    # Final residual value should be between 20% and 95%
    assert 20.0 <= traj[-1]["residual_value_pct"] <= 95.0
    # Cumulative depreciation should increase
    assert traj[-1]["cumulative_depreciation_egp"] >= traj[0]["cumulative_depreciation_egp"]


def test_forecast_depreciation_summary_structure(forecaster: DepreciationForecaster) -> None:
    """Test summary dictionary fields and retention tiers."""
    result = forecaster.forecast_depreciation(
        make="Kia",
        model="Sportage",
        year=2022,
        mileage=35_000,
    )
    summary = result["summary"]
    assert "five_year_residual_pct" in summary
    assert "five_year_depreciation_pct" in summary
    assert "average_annual_depreciation_pct" in summary
    assert "retention_tier" in summary
    assert isinstance(summary["retention_tier"], str)


def test_depreciation_steep_decay_tier() -> None:
    """Test retention tier classification when residual value is low."""
    class RapidDecayEngine:
        def predict_valuation(self, **kwargs):
            return {
                "fair_market_value": 1000000.0,
                "quantiles": {"mean": 1000000.0, "p10": 800000.0, "p90": 1200000.0},
            }
        def predict_quantiles(self, df):
            return {"mean": 300000.0, "p10": 200000.0, "p90": 400000.0}

    decay_forecaster = DepreciationForecaster(valuation_engine=RapidDecayEngine())
    res = decay_forecaster.forecast_depreciation("FakeMake", "FakeModel", 2020, 50000, years_ahead=5)
    assert res["summary"]["five_year_residual_pct"] <= 40.0
    assert "Steep Depreciation" in res["summary"]["retention_tier"]

