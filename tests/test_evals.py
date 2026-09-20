"""Unit tests for the evaluation harness and benchmark runner."""

from pathlib import Path
import pytest

from evals.runner import (
    evaluate_dataset,
    benchmark_depreciation,
    benchmark_latency,
    generate_markdown_report,
    run_eval_suite,
)


def test_evaluate_dataset_valid():
    """Verify evaluation on authentic sample dataset produces valid regression and quantile metrics."""
    res = evaluate_dataset()
    assert res["num_samples"] >= 40
    assert 0.80 <= res["r2_score"] <= 1.0
    assert res["mae_egp"] > 0
    assert res["mape_pct"] > 0
    assert res["picp_80_coverage_pct"] >= 70.0
    assert "deal_ratings_distribution" in res
    assert res["mean_deal_score"] >= 0.0


def test_evaluate_dataset_missing_file():
    """Verify FileNotFoundError is raised when pointing to an invalid data path."""
    with pytest.raises(FileNotFoundError):
        evaluate_dataset("nonexistent_cars_dataset.csv")


def test_benchmark_depreciation():
    """Verify depreciation evaluation generates trajectories across standard archetypes."""
    res = benchmark_depreciation()
    assert len(res) == 5
    for item in res:
        assert "vehicle" in item
        assert item["starting_price_egp"] > 0
        assert item["year_5_price_egp"] > 0
        assert 0.0 < item["residual_value_5yr_pct"] <= 100.0
        assert item["starting_price_egp"] >= item["year_5_price_egp"]
        assert "retention_tier" in item


def test_benchmark_latency():
    """Verify inference latency benchmark executes and reports timing metrics."""
    res = benchmark_latency(n_iterations=10)
    assert res["mean_latency_ms"] > 0.0
    assert res["median_latency_ms"] > 0.0
    assert res["p95_latency_ms"] >= res["median_latency_ms"]
    assert res["throughput_req_per_sec"] > 0.0


def test_generate_markdown_report(tmp_path):
    """Verify markdown report generation correctly outputs markdown with key sections."""
    sample_results = {
        "dataset_evaluation": {
            "num_samples": 45,
            "r2_score": 0.9576,
            "mae_egp": 89268.0,
            "rmse_egp": 120000.0,
            "mape_pct": 17.24,
            "mdape_pct": 5.74,
            "picp_80_coverage_pct": 88.89,
            "picp_50_coverage_pct": 60.0,
            "mean_spread_egp": 368319.0,
            "mean_spread_pct": 41.9,
            "deal_ratings_distribution": {"Fair Deal": 21, "High Price": 20},
            "deal_ratings_distribution_pct": {"Fair Deal": 46.7, "High Price": 44.4},
            "mean_deal_score": 39.8,
        },
        "depreciation_benchmarks": [
            {
                "id": "LUXURY_SUV",
                "segment": "Luxury Compact SUV",
                "vehicle": "BMW X1 (2020)",
                "starting_price_egp": 3000000.0,
                "year_5_price_egp": 1600000.0,
                "residual_value_5yr_pct": 53.2,
                "annual_depreciation_pct": 11.6,
                "retention_tier": "Moderate Value Retention",
            }
        ],
        "latency_benchmark": {
            "mean_latency_ms": 8.35,
            "median_latency_ms": 8.16,
            "p95_latency_ms": 9.43,
            "p99_latency_ms": 10.66,
            "throughput_req_per_sec": 120.0,
        },
    }

    out_file = tmp_path / "test_benchmark.md"
    generate_markdown_report(sample_results, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Executive Performance Matrix" in content
    assert "Quantile Calibration" in content
    assert "Fair Market Deal Rating Distribution" in content
    assert "BMW X1" in content


def test_run_eval_suite(tmp_path):
    """Verify run_eval_suite executes end-to-end and outputs file."""
    rep_file = tmp_path / "custom_eval.md"
    res = run_eval_suite(report_path=rep_file)
    assert rep_file.exists()
    assert res["r2_score"] > 0.8
    assert "dataset_evaluation" in res
    assert "depreciation_benchmarks" in res
    assert "latency_benchmark" in res
