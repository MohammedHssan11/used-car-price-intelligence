"""Unit tests for the used car price intelligence CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if "cli" in sys.modules and not getattr(sys.modules["cli"], "__file__", "").startswith(str(ROOT_DIR)):
    del sys.modules["cli"]

from cli import build_parser, main


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Test calling CLI with no arguments displays usage."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "car-pricing-cli" in captured.out


def test_cli_predict(capsys: pytest.CaptureFixture) -> None:
    """Test predict subcommand."""
    ret = main(["predict", "--make", "Toyota", "--model", "Corolla", "--year", "2020", "--mileage", "60000", "--verbose"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "USED CAR VALUATION REPORT" in captured.out
    assert "Fair Market Value:" in captured.out
    assert "Full Quantile Price Distribution:" in captured.out


def test_cli_predict_json(capsys: pytest.CaptureFixture) -> None:
    """Test predict subcommand outputting JSON."""
    ret = main(["predict", "--make", "Hyundai", "--model", "Tucson Turbo GDI", "--year", "2021", "--mileage", "75000", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "fair_market_value_egp" in data
    assert "quantile_bands" in data
    assert data["fair_market_value_egp"] > 0


def test_cli_deal_check(capsys: pytest.CaptureFixture) -> None:
    """Test deal-check subcommand."""
    ret = main([
        "deal-check",
        "--asking-price", "1200000",
        "--make", "Hyundai",
        "--model", "Tucson Turbo GDI",
        "--year", "2021",
        "--mileage", "80000",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "FAIR MARKET DEAL ADVISOR" in captured.out
    assert "Deal Assessment:" in captured.out


def test_cli_deal_check_json(capsys: pytest.CaptureFixture) -> None:
    """Test deal-check subcommand outputting JSON."""
    ret = main([
        "deal-check",
        "--asking-price", "1800000",
        "--make", "BMW",
        "--model", "X1",
        "--year", "2018",
        "--mileage", "84000",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "deal_rating" in data
    assert "deal_score" in data


def test_cli_depreciation(capsys: pytest.CaptureFixture) -> None:
    """Test depreciation subcommand."""
    ret = main([
        "depreciation",
        "--make", "Kia",
        "--model", "Sportage",
        "--year", "2022",
        "--mileage", "40000",
        "--years-ahead", "3",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "5-YEAR DEPRECIATION TRAJECTORY" in captured.out or "DEPRECIATION TRAJECTORY" in captured.out
    assert "Residual %" in captured.out


def test_cli_depreciation_json(capsys: pytest.CaptureFixture) -> None:
    """Test depreciation subcommand outputting JSON."""
    ret = main([
        "depreciation",
        "--make", "Kia",
        "--model", "Sportage",
        "--year", "2022",
        "--mileage", "40000",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "trajectory" in data
    assert "summary" in data


def test_cli_batch(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test batch appraisal of sample CSV."""
    in_csv = ROOT_DIR / "data" / "sample_used_cars.csv"
    out_json = tmp_path / "appraised.json"
    ret = main(["batch", "--input", str(in_csv), "--limit", "5", "--output", str(out_json), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["processed_vehicles"] == 5
    assert out_json.exists()


def test_cli_batch_missing_file(capsys: pytest.CaptureFixture) -> None:
    """Test batch error on nonexistent input file."""
    ret = main(["batch", "--input", "non_existent.csv"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error: Input file not found" in captured.err


def test_cli_benchmark(capsys: pytest.CaptureFixture) -> None:
    """Test benchmark subcommand."""
    ret = main(["benchmark", "--samples", "5", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "mean_latency_ms" in data
    assert data["evaluated_vehicles"] == 5


def test_cli_benchmark_text_mode(capsys: pytest.CaptureFixture) -> None:
    """Test benchmark subcommand in plain text mode."""
    ret = main(["benchmark", "--samples", "3"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "VALUATION ENGINE LATENCY BENCHMARK" in captured.out
    assert "Mean Inference Latency" in captured.out


def test_cli_empty_args(capsys: pytest.CaptureFixture) -> None:
    """Test running CLI without arguments prints help and exits 0."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "used car" in captured.out.lower()

