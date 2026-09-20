"""Unit tests for automotive data preprocessing and feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessor import (
    REFERENCE_YEAR,
    create_vehicle_dataframe,
    engineer_features,
    parse_mileage,
    parse_price,
    parse_year_from_name,
)


def test_parse_price_formats() -> None:
    """Test price parsing from various currency string formats."""
    assert parse_price("2,800,000 EGP") == 2_800_000.0
    assert parse_price("1,200,000") == 1_200_000.0
    assert parse_price(450_000) == 450_000.0
    assert parse_price(None) is None
    assert parse_price("") is None
    assert parse_price("Call for price") is None


def test_parse_mileage_formats() -> None:
    """Test mileage extraction from formatted strings."""
    assert parse_mileage("86,000 Km") == 86_000.0
    assert parse_mileage("300 Km") == 300.0
    assert parse_mileage(120_000) == 120_000.0
    assert parse_mileage(None) is None
    assert parse_mileage("New") is None


def test_parse_year_from_name() -> None:
    """Test extracting manufacture year from vehicle title."""
    assert parse_year_from_name("Kia Sportage 2024") == 2024
    assert parse_year_from_name("Fiat Tipo 2021") == 2021
    assert parse_year_from_name("BMW X1 2018") == 2018
    assert parse_year_from_name("Vintage Beetle 1974") == 1974
    assert parse_year_from_name("Car Without Year") is None


def test_engineer_features() -> None:
    """Test vehicle age and annual mileage calculations."""
    age, mileage_yr = engineer_features(year=2020, mileage=80_000, reference_year=2024)
    assert age == 4
    assert mileage_yr == 20_000.0

    # Test brand new car (age 0 -> effective 1)
    age_new, mileage_new = engineer_features(year=2024, mileage=5_000, reference_year=2024)
    assert age_new == 0
    assert mileage_new == 5_000.0


def test_create_vehicle_dataframe() -> None:
    """Test DataFrame assembly for model ingestion."""
    df = create_vehicle_dataframe(
        make="Hyundai",
        model="Tucson Turbo GDI",
        year=2021,
        mileage=60_000,
        color="Black",
        city="Cairo",
    )
    assert len(df) == 1
    expected_cols = [
        "mileage",
        "car_age",
        "mileage_per_year",
        "Make",
        "Model",
        "Color",
        "City",
        "Automatic Transmission",
        "Air Conditioner",
        "Power Steering",
        "Remote Control",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["Make"].iloc[0] == "Hyundai"
    assert df["car_age"].iloc[0] == REFERENCE_YEAR - 2021
