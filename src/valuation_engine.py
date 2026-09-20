"""Automotive valuation engine with ensemble quantile regression uncertainty bands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.preprocessor import create_vehicle_dataframe


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"


class CarValuationEngine:
    """Automotive valuation model with ensemble quantile regression uncertainty bands."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, pipeline: Any = "DEFAULT"):
        if pipeline != "DEFAULT":
            self.pipeline = pipeline
        else:
            p_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            if p_path.exists():
                self.pipeline = joblib.load(p_path)
            else:
                self.pipeline = None

        self.preprocessor = getattr(self.pipeline, "named_steps", {}).get("preprocessor") if self.pipeline else None
        self.regressor = getattr(self.pipeline, "named_steps", {}).get("model") if self.pipeline else None

    def predict_point_estimate(self, car_df: pd.DataFrame) -> float:
        """Predict expected mean listing price in EGP."""
        if self.pipeline is None or car_df is None or car_df.empty:
            return 0.0
        val = float(self.pipeline.predict(car_df)[0])
        return max(0.0, round(val, 2))

    def predict_quantiles(
        self,
        car_df: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Compute ensemble quantile regression bands across all Random Forest estimators.

        Quantiles:
        - P10: Bargain / Lower bound (10th percentile)
        - P25: Competitive entry price (25th percentile)
        - P50: Median fair market value (50th percentile)
        - P75: Upper fair market range (75th percentile)
        - P90: Dealer premium ceiling (90th percentile)

        Returns:
            Dictionary of quantile estimates, standard deviation, and interval width.
        """
        if self.pipeline is None or car_df is None or car_df.empty:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "interval_80": 0.0,
                "uncertainty_pct": 0.0,
            }

        # Check if regressor has forest estimators
        estimators = getattr(self.regressor, "estimators_", None) if self.regressor is not None else None
        if estimators is not None and len(estimators) > 0 and self.preprocessor is not None:
            X_trans = self.preprocessor.transform(car_df)
            tree_preds = np.array([float(tree.predict(X_trans)[0]) for tree in estimators])
            tree_preds = np.clip(tree_preds, 0.0, None)

            mean_val = float(np.mean(tree_preds))
            std_val = float(np.std(tree_preds))
            p10_val = float(np.percentile(tree_preds, 10))
            p25_val = float(np.percentile(tree_preds, 25))
            p50_val = float(np.percentile(tree_preds, 50))
            p75_val = float(np.percentile(tree_preds, 75))
            p90_val = float(np.percentile(tree_preds, 90))
        else:
            # Fallback for non-ensemble models: use point estimate with synthetic spread
            point = self.predict_point_estimate(car_df)
            mean_val = point
            std_val = point * 0.10
            p10_val = point * 0.85
            p25_val = point * 0.92
            p50_val = point
            p75_val = point * 1.08
            p90_val = point * 1.15

        interval_80 = max(0.0, p90_val - p10_val)
        uncertainty_pct = (std_val / max(1.0, mean_val)) * 100.0

        return {
            "mean": round(mean_val, 2),
            "median": round(p50_val, 2),
            "std": round(std_val, 2),
            "p10": round(p10_val, 2),
            "p25": round(p25_val, 2),
            "p50": round(p50_val, 2),
            "p75": round(p75_val, 2),
            "p90": round(p90_val, 2),
            "interval_80": round(interval_80, 2),
            "uncertainty_pct": round(uncertainty_pct, 2),
        }

    def predict_valuation(
        self,
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
        """Execute full vehicle valuation returning point estimate, quantile bands, and specifications."""
        car_df = create_vehicle_dataframe(
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

        quantiles = self.predict_quantiles(car_df)
        point_estimate = quantiles["mean"]

        return {
            "vehicle": {
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "car_age": int(car_df["car_age"].iloc[0]),
                "mileage_per_year": float(car_df["mileage_per_year"].iloc[0]),
                "color": color,
                "city": city,
                "automatic": automatic_transmission,
            },
            "fair_market_value": point_estimate,
            "quantiles": quantiles,
            "car_df": car_df,
        }
