"""Automotive data preprocessing, string parsing, and feature engineering engine."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


REFERENCE_YEAR = 2024

RE_PRICE = re.compile(r"(\d+(?:,\d+)*)")
RE_YEAR = re.compile(r"((?:19|20)\d{2})(?!.*(?:19|20)\d{2})")


def parse_price(value: Any) -> Optional[float]:
    """Parse numeric price from formatted strings (e.g. '2,800,000 EGP', '1,200,000')."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None

    s = str(value).strip()
    match = RE_PRICE.search(s)
    if match:
        clean_val = match.group(1).replace(",", "")
        try:
            return float(clean_val)
        except ValueError:
            return None
    return None


def parse_mileage(value: Any) -> Optional[float]:
    """Parse numeric mileage from strings (e.g. '86,000 Km', '131,000')."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None

    s = str(value).strip()
    match = RE_PRICE.search(s)
    if match:
        clean_val = match.group(1).replace(",", "")
        try:
            return float(clean_val)
        except ValueError:
            return None
    return None


def parse_year_from_name(name_str: Any) -> Optional[int]:
    """Extract 4-digit model year from vehicle listing title (e.g. 'Kia Sportage 2024')."""
    if name_str is None or pd.isna(name_str):
        return None
    s = str(name_str).strip()
    match = RE_YEAR.search(s)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def engineer_features(
    year: int,
    mileage: float,
    reference_year: int = REFERENCE_YEAR,
) -> Tuple[int, float]:
    """Calculate vehicle age and annual mileage intensity.

    Returns:
        Tuple of (car_age, mileage_per_year)
    """
    car_age = max(0, reference_year - int(year))
    effective_years = max(car_age, 1)
    mileage_per_year = float(mileage) / float(effective_years)
    return car_age, round(mileage_per_year, 2)


def create_vehicle_dataframe(
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
    reference_year: int = REFERENCE_YEAR,
) -> pd.DataFrame:
    """Assemble a standardized single-row DataFrame formatted for the model pipeline."""
    car_age, mileage_per_year = engineer_features(year, mileage, reference_year=reference_year)

    data = {
        "mileage": [float(mileage)],
        "car_age": [int(car_age)],
        "mileage_per_year": [float(mileage_per_year)],
        "Make": [str(make).strip()],
        "Model": [str(model).strip()],
        "Color": [str(color).strip()],
        "City": [str(city).strip()],
        "Automatic Transmission": ["Yes" if str(automatic_transmission).lower() in ["yes", "true", "1"] else "No"],
        "Air Conditioner": ["Yes" if str(air_conditioner).lower() in ["yes", "true", "1"] else "No"],
        "Power Steering": ["Yes" if str(power_steering).lower() in ["yes", "true", "1"] else "No"],
        "Remote Control": ["Yes" if str(remote_control).lower() in ["yes", "true", "1"] else "No"],
    }
    return pd.DataFrame(data)
