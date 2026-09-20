from pathlib import Path
import joblib
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_YEAR = 2024


def test_sample_data_exists():
    sample_path = PROJECT_DIR / "data" / "sample_used_cars.csv"
    assert sample_path.exists()
    df = pd.read_csv(sample_path)
    assert len(df) > 0
    assert "Make" in df.columns
    assert "Model" in df.columns


def test_feature_engineering_calculation():
    year = 2018
    mileage = 120_000
    car_age = REFERENCE_YEAR - year
    mileage_per_year = mileage / max(car_age, 1)

    assert car_age == 6
    assert mileage_per_year == 20_000.0


def test_preprocessor_artifact():
    preprocessor_path = PROJECT_DIR / "models" / "preprocessor.pkl"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        assert preprocessor is not None


def test_feature_importance_format():
    fi_path = PROJECT_DIR / "models" / "feature_importance.csv"
    if fi_path.exists():
        df_fi = pd.read_csv(fi_path)
        assert len(df_fi) > 0
