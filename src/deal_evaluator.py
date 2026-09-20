"""Automotive fair market deal rating and buyer negotiation advisor engine."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.valuation_engine import CarValuationEngine


RATING_GREAT_DEAL = "Great Deal"
RATING_GOOD_DEAL = "Good Deal"
RATING_FAIR_DEAL = "Fair Deal"
RATING_HIGH_PRICE = "High Price"
RATING_OVERPRICED = "Overpriced"


class DealEvaluator:
    """Evaluates an advertised used car asking price against empirical quantile market distributions."""

    def __init__(self, valuation_engine: Optional[CarValuationEngine] = None):
        self.engine = valuation_engine or CarValuationEngine()

    def evaluate_deal(
        self,
        asking_price: float,
        make: str,
        model: str,
        year: int,
        mileage: float,
        color: str = "Black",
        city: str = "Cairo",
        automatic_transmission: str = "Yes",
        air_conditioner: str = "Yes",
        power_steering: str = "Yes",
        remote_control: str = "Yes",
    ) -> Dict[str, Any]:
        """Benchmark asking price against fair market valuation and quantile distribution.

        Returns:
            Dictionary containing deal rating, deal score (0-100), savings/premium, and negotiation advice.
        """
        val_result = self.engine.predict_valuation(
            make=make,
            model=model,
            year=year,
            mileage=mileage,
            color=color,
            city=city,
            automatic_transmission=automatic_transmission,
            air_conditioner=air_conditioner,
            power_steering=power_steering,
            remote_control=remote_control,
        )

        quantiles = val_result["quantiles"]
        median_price = max(1.0, quantiles["p50"])
        p10 = quantiles["p10"]
        p25 = quantiles["p25"]
        p75 = quantiles["p75"]
        p90 = quantiles["p90"]

        # Difference: Positive = Buyer Savings, Negative = Buyer Overpayment
        delta_egp = median_price - asking_price
        delta_pct = (delta_egp / median_price) * 100.0

        # Classification based on quantiles and percentage margins
        if asking_price <= p10 or delta_pct >= 12.0:
            rating = RATING_GREAT_DEAL
            badge_icon = "🟢"
            deal_score = min(100, round(85.0 + max(0.0, delta_pct - 12.0) * 1.5, 1))
            advice = f"Strong purchase opportunity! Priced EGP {abs(delta_egp):,.0f} ({abs(delta_pct):.1f}%) below fair market median. Immediate inspection recommended."

        elif asking_price <= p25 or delta_pct >= 4.0:
            rating = RATING_GOOD_DEAL
            badge_icon = "🟢"
            deal_score = round(70.0 + (delta_pct - 4.0) * 1.8, 1)
            advice = f"Good value. Priced EGP {abs(delta_egp):,.0f} ({abs(delta_pct):.1f}%) below prevailing dealership averages."

        elif asking_price <= p75 and abs(delta_pct) < 6.0:
            rating = RATING_FAIR_DEAL
            badge_icon = "🔵"
            deal_score = round(50.0 + delta_pct * 2.0, 1)
            advice = "Accurately priced at fair market listing value. Verify vehicle maintenance history before committing."

        elif asking_price <= p90 or delta_pct > -15.0:
            rating = RATING_HIGH_PRICE
            badge_icon = "🟡"
            deal_score = max(10, round(45.0 + delta_pct * 1.5, 1))
            advice = f"Priced higher than average by EGP {abs(delta_egp):,.0f} ({abs(delta_pct):.1f}% premium). Target a negotiation discount towards EGP {median_price:,.0f}."

        else:
            rating = RATING_OVERPRICED
            badge_icon = "🔴"
            deal_score = max(0, round(25.0 + delta_pct, 1))
            advice = f"Substantially overpriced by EGP {abs(delta_egp):,.0f} ({abs(delta_pct):.1f}% above market ceiling). Consider alternative listings unless significant premium options are verified."

        return {
            "asking_price": asking_price,
            "fair_market_median": median_price,
            "quantiles": quantiles,
            "deal_rating": rating,
            "badge": f"{badge_icon} {rating}",
            "deal_score": deal_score,
            "price_difference_egp": round(delta_egp, 2),
            "price_difference_pct": round(delta_pct, 2),
            "is_savings": delta_egp >= 0,
            "negotiation_advice": advice,
            "vehicle": val_result["vehicle"],
        }
