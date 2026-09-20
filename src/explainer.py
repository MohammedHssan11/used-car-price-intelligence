"""Automotive valuation explainability, feature attribution, and market driver analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.preprocessor import REFERENCE_YEAR


LUXURY_BRANDS = {"BMW", "Mercedes-Benz", "Audi", "Porsche", "Land Rover", "Jaguar", "Volvo", "Lexus", "Jeep"}
POPULAR_RELIABLE_BRANDS = {"Toyota", "Honda", "Volkswagen", "Subaru", "Mazda", "Skoda"}
ECONOMY_VOLUME_BRANDS = {"Hyundai", "Kia", "Nissan", "Fiat", "Renault", "Chevrolet", "Chery", "MG", "BYD", "Geely"}


class ValuationExplainer:
    """Provides transparent, human-readable explanations of vehicle price drivers."""

    def explain_valuation(
        self,
        make: str,
        model: str,
        year: int,
        mileage: float,
        fair_price: float,
        automatic_transmission: str = "Yes",
        city: str = "Cairo",
    ) -> Dict[str, Any]:
        """Generate qualitative and quantitative feature attribution insights."""
        car_age = max(0, REFERENCE_YEAR - year)
        mileage_per_year = mileage / max(car_age, 1)

        drivers: List[Dict[str, str]] = []

        # 1. Age Driver
        if car_age <= 2:
            drivers.append(
                {
                    "feature": "Vehicle Age",
                    "impact": "Strong Positive (+)",
                    "detail": f"Late-model vehicle ({car_age} years old); retains high factory warranty and modern styling value.",
                }
            )
        elif car_age <= 7:
            drivers.append(
                {
                    "feature": "Vehicle Age",
                    "impact": "Neutral / Moderate (-)",
                    "detail": f"Typical secondary market age ({car_age} years old); standard mid-lifecycle depreciation applied.",
                }
            )
        else:
            drivers.append(
                {
                    "feature": "Vehicle Age",
                    "impact": "Negative (-)",
                    "detail": f"Older generation vehicle ({car_age} years old); higher ongoing maintenance reserve required.",
                }
            )

        # 2. Mileage Driver
        if mileage_per_year < 12_000:
            drivers.append(
                {
                    "feature": "Mileage Usage",
                    "impact": "Positive (+)",
                    "detail": f"Low annual mileage ({mileage_per_year:,.0f} km/yr vs 18k market avg); lower mechanical wear commands premium.",
                }
            )
        elif mileage_per_year <= 22_000:
            drivers.append(
                {
                    "feature": "Mileage Usage",
                    "impact": "Neutral",
                    "detail": f"Standard commuter usage ({mileage_per_year:,.0f} km/yr); aligned with national driving averages.",
                }
            )
        else:
            drivers.append(
                {
                    "feature": "Mileage Usage",
                    "impact": "Negative (-)",
                    "detail": f"High annual usage intensity ({mileage_per_year:,.0f} km/yr); reflects commercial or heavy highway commute wear.",
                }
            )

        # 3. Brand Prestige & Market Liquidity
        if make in LUXURY_BRANDS:
            brand_tier = "Luxury European / Premium"
            brand_impact = "High Market Value / High Parts Cost"
            brand_detail = f"{make} commands premium brand prestige, balanced against higher parts and maintenance costs."
        elif make in POPULAR_RELIABLE_BRANDS:
            brand_tier = "High Resale / Japanese-German Volume"
            brand_impact = "Exceptional Resale Liquidity (+)"
            brand_detail = f"{make} benefits from strong Egyptian secondary market liquidity and renowned mechanical durability."
        else:
            brand_tier = "High-Volume Accessible"
            brand_impact = "Balanced Resale / Affordable Maintenance"
            brand_detail = f"{make} provides accessible ownership with wide parts availability and rapid market turnover."

        drivers.append(
            {
                "feature": "Brand Tier & Liquidity",
                "impact": brand_impact,
                "detail": brand_detail,
            }
        )

        # 4. Transmission
        if str(automatic_transmission).lower() in ["yes", "true", "1"]:
            drivers.append(
                {
                    "feature": "Transmission",
                    "impact": "Positive (+)",
                    "detail": "Automatic transmission commands strong preference in congested urban Egyptian traffic.",
                }
            )
        else:
            drivers.append(
                {
                    "feature": "Transmission",
                    "impact": "Discount (-)",
                    "detail": "Manual transmission appeals to niche economy or driving enthusiast buyers at discounted valuation.",
                }
            )

        summary_text = (
            f"Estimated at EGP {fair_price:,.0f} based on {car_age}-year age ({year} model) and "
            f"{mileage:,.0f} km total distance ({mileage_per_year:,.0f} km/yr). "
            f"Valuation reflects {make} ({brand_tier}) brand positioning and local secondary market liquidity."
        )

        return {
            "summary_rationale": summary_text,
            "drivers": drivers,
            "brand_tier": brand_tier,
            "mileage_intensity": f"{mileage_per_year:,.0f} km/year",
            "car_age_years": car_age,
        }
