"""Multi-year automotive depreciation forecasting and residual value projection engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.preprocessor import REFERENCE_YEAR, create_vehicle_dataframe
from src.valuation_engine import CarValuationEngine


class DepreciationForecaster:
    """Projects future vehicle valuation trajectories and residual values over a 1 to 5-year horizon."""

    def __init__(self, valuation_engine: Optional[CarValuationEngine] = None):
        self.engine = valuation_engine or CarValuationEngine()

    def forecast_depreciation(
        self,
        make: str,
        model: str,
        year: int,
        mileage: float,
        annual_mileage: float = 15_000.0,
        years_ahead: int = 5,
        color: str = "Black",
        city: str = "Cairo",
        automatic_transmission: str = "Yes",
        air_conditioner: str = "Yes",
        power_steering: str = "Yes",
        remote_control: str = "Yes",
    ) -> Dict[str, Any]:
        """Simulate future vehicle valuation for each year up to `years_ahead`.

        Returns:
            Dictionary containing annual trajectory points, residual values, and depreciation summary.
        """
        # Baseline Year 0 (Current state)
        base_val = self.engine.predict_valuation(
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
        current_price = max(1.0, base_val["fair_market_value"])

        trajectory: List[Dict[str, Any]] = [
            {
                "year_offset": 0,
                "calendar_year": REFERENCE_YEAR,
                "projected_mileage": round(mileage, 0),
                "estimated_price": round(current_price, 2),
                "p10_price": base_val["quantiles"]["p10"],
                "p90_price": base_val["quantiles"]["p90"],
                "cumulative_depreciation_egp": 0.0,
                "cumulative_depreciation_pct": 0.0,
                "annual_depreciation_pct": 0.0,
                "residual_value_pct": 100.0,
            }
        ]

        prev_price = current_price

        for t in range(1, years_ahead + 1):
            future_calendar_year = REFERENCE_YEAR + t
            simulated_mileage = mileage + (annual_mileage * t)
            # The model was manufactured in `year`, so its age in calendar year `future_calendar_year` is:
            simulated_age = (REFERENCE_YEAR + t) - year

            future_df = create_vehicle_dataframe(
                make=make,
                model=model,
                year=year,
                mileage=simulated_mileage,
                color=color,
                city=city,
                automatic_transmission=automatic_transmission,
                air_conditioner=air_conditioner,
                power_steering=power_steering,
                remote_control=remote_control,
                reference_year=future_calendar_year,
            )

            future_quantiles = self.engine.predict_quantiles(future_df)
            future_price = max(0.0, future_quantiles["mean"])

            # Ensure logical monotonic non-increasing property for depreciation simulation
            if future_price > prev_price:
                # In rare edge cases where tree boundary increases with age/mileage, apply standard 8% decay
                future_price = prev_price * 0.92

            cum_dep_egp = max(0.0, current_price - future_price)
            cum_dep_pct = (cum_dep_egp / current_price) * 100.0
            annual_dep_pct = max(0.0, (prev_price - future_price) / max(1.0, prev_price)) * 100.0
            residual_pct = max(0.0, (future_price / current_price) * 100.0)

            trajectory.append(
                {
                    "year_offset": t,
                    "calendar_year": future_calendar_year,
                    "projected_mileage": round(simulated_mileage, 0),
                    "estimated_price": round(future_price, 2),
                    "p10_price": round(min(future_price, future_quantiles["p10"]), 2),
                    "p90_price": round(max(future_price, future_quantiles["p90"]), 2),
                    "cumulative_depreciation_egp": round(cum_dep_egp, 2),
                    "cumulative_depreciation_pct": round(cum_dep_pct, 1),
                    "annual_depreciation_pct": round(annual_dep_pct, 1),
                    "residual_value_pct": round(residual_pct, 1),
                }
            )
            prev_price = future_price

        # Summary Metrics
        final_residual = trajectory[-1]["residual_value_pct"]
        if final_residual >= 65.0:
            retention_tier = "High Value Retention (Class Leader)"
        elif final_residual >= 45.0:
            retention_tier = "Moderate Value Retention (Market Average)"
        else:
            retention_tier = "Steep Depreciation (High Depreciation Rate)"

        avg_annual_decay = round(float(np.mean([pt["annual_depreciation_pct"] for pt in trajectory[1:]])), 1)

        return {
            "current_fair_price": current_price,
            "annual_mileage_assumption": annual_mileage,
            "years_forecasted": years_ahead,
            "trajectory": trajectory,
            "summary": {
                "five_year_residual_pct": final_residual,
                "five_year_depreciation_pct": round(100.0 - final_residual, 1),
                "five_year_loss_egp": trajectory[-1]["cumulative_depreciation_egp"],
                "average_annual_depreciation_pct": avg_annual_decay,
                "retention_tier": retention_tier,
            },
        }
