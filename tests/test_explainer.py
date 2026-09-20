"""Unit tests for automotive valuation explainability and feature attribution."""

from __future__ import annotations

import pytest
from src.explainer import ValuationExplainer


@pytest.fixture(scope="module")
def explainer() -> ValuationExplainer:
    return ValuationExplainer()


def test_explainer_luxury_brand(explainer: ValuationExplainer) -> None:
    """Test brand prestige categorization for luxury vehicles."""
    res = explainer.explain_valuation(
        make="BMW",
        model="X1",
        year=2021,
        mileage=50_000,
        fair_price=2_000_000.0,
    )
    assert "Luxury" in res["brand_tier"]
    assert "BMW" in res["summary_rationale"]
    assert len(res["drivers"]) >= 3


def test_explainer_mileage_wear(explainer: ValuationExplainer) -> None:
    """Test high vs low mileage usage detection."""
    # Low mileage: 15,000 km over 3 years = 5,000 km/yr (Positive)
    res_low = explainer.explain_valuation(
        make="Toyota",
        model="Corolla",
        year=2021,
        mileage=15_000,
        fair_price=1_100_000.0,
    )
    mileage_driver_low = [d for d in res_low["drivers"] if d["feature"] == "Mileage Usage"][0]
    assert "Positive" in mileage_driver_low["impact"]

    # High mileage: 120,000 km over 3 years = 40,000 km/yr (Negative)
    res_high = explainer.explain_valuation(
        make="Toyota",
        model="Corolla",
        year=2021,
        mileage=120_000,
        fair_price=800_000.0,
    )
    mileage_driver_high = [d for d in res_high["drivers"] if d["feature"] == "Mileage Usage"][0]
    assert "Negative" in mileage_driver_high["impact"]
