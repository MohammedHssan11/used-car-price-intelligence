"""Unit tests for automotive deal rating and buyer negotiation advisor."""

from __future__ import annotations

import pytest
from src.deal_evaluator import (
    DealEvaluator,
    RATING_FAIR_DEAL,
    RATING_GOOD_DEAL,
    RATING_GREAT_DEAL,
    RATING_HIGH_PRICE,
    RATING_OVERPRICED,
)
from src.valuation_engine import CarValuationEngine


@pytest.fixture(scope="module")
def evaluator() -> DealEvaluator:
    engine = CarValuationEngine()
    return DealEvaluator(valuation_engine=engine)


def test_deal_evaluator_great_deal(evaluator: DealEvaluator) -> None:
    """Test deal rating when vehicle asking price is heavily discounted."""
    # Market price for 2021 Tucson is around 1.5M - 1.8M EGP; asking 900k should be Great Deal
    res = evaluator.evaluate_deal(
        asking_price=900_000.0,
        make="Hyundai",
        model="Tucson Turbo GDI",
        year=2021,
        mileage=80_000,
    )
    assert res["deal_rating"] in [RATING_GREAT_DEAL, RATING_GOOD_DEAL]
    assert res["is_savings"] is True
    assert res["price_difference_egp"] > 0
    assert res["deal_score"] >= 70.0


def test_deal_evaluator_overpriced(evaluator: DealEvaluator) -> None:
    """Test deal rating when vehicle asking price is excessive."""
    # Asking 3.5M EGP for a 2018 X1 should be Overpriced
    res = evaluator.evaluate_deal(
        asking_price=3_500_000.0,
        make="BMW",
        model="X1",
        year=2018,
        mileage=120_000,
    )
    assert res["deal_rating"] in [RATING_OVERPRICED, RATING_HIGH_PRICE]
    assert res["is_savings"] is False
    assert res["price_difference_egp"] < 0
    assert res["deal_score"] <= 45.0


def test_deal_evaluator_fair_deal(evaluator: DealEvaluator) -> None:
    """Test deal rating when asking price matches fair market value exactly."""
    val = evaluator.engine.predict_valuation(
        make="Toyota",
        model="Corolla",
        year=2020,
        mileage=60_000,
    )
    median_price = val["quantiles"]["p50"]

    res = evaluator.evaluate_deal(
        asking_price=median_price,
        make="Toyota",
        model="Corolla",
        year=2020,
        mileage=60_000,
    )
    assert res["deal_rating"] == RATING_FAIR_DEAL
    assert abs(res["price_difference_pct"]) < 1.0
    assert "Accurately priced" in res["negotiation_advice"]
