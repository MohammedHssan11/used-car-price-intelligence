"""Enterprise Command-Line Interface (CLI) for Used Car Price Intelligence.

Subcommands:
- predict: Estimate fair market valuation with ensemble quantile regression bands.
- deal-check: Assess whether an advertised asking price is a Great, Good, Fair, or Overpriced deal.
- depreciation: Forecast 5-year future depreciation trajectory and residual value retention.
- batch: Bulk process vehicle listings from a CSV file.
- benchmark: Measure valuation latency and model accuracy diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.deal_evaluator import DealEvaluator
from src.depreciation_forecaster import DepreciationForecaster
from src.explainer import ValuationExplainer
from src.preprocessor import parse_mileage, parse_price, parse_year_from_name
from src.valuation_engine import CarValuationEngine


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the automotive CLI."""
    parser = argparse.ArgumentParser(
        prog="car-pricing-cli",
        description="Enterprise Automotive Valuation, Quantile Uncertainty & Depreciation CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. predict
    p_pred = subparsers.add_parser("predict", help="Estimate vehicle market price with quantile uncertainty")
    p_pred.add_argument("--make", type=str, required=True, help="Manufacturer brand (e.g. Toyota, BMW, Hyundai)")
    p_pred.add_argument("--model", type=str, required=True, help="Vehicle model name (e.g. Corolla, Sportage)")
    p_pred.add_argument("--year", type=int, required=True, help="Model year of manufacture (e.g. 2021)")
    p_pred.add_argument("--mileage", type=float, required=True, help="Odometer distance in kilometers")
    p_pred.add_argument("--color", type=str, default="Black", help="Vehicle exterior paint color")
    p_pred.add_argument("--city", type=str, default="Cairo", help="Listing geographical location")
    p_pred.add_argument("--transmission", type=str, default="Yes", choices=["Yes", "No"], help="Automatic transmission (Yes/No)")
    p_pred.add_argument("--verbose", action="store_true", help="Display full quantile spread and feature drivers")
    p_pred.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 2. deal-check
    p_deal = subparsers.add_parser("deal-check", help="Evaluate whether an asking price is a good deal")
    p_deal.add_argument("--asking-price", type=float, required=True, help="Advertised asking listing price in EGP")
    p_deal.add_argument("--make", type=str, required=True, help="Manufacturer brand")
    p_deal.add_argument("--model", type=str, required=True, help="Vehicle model name")
    p_deal.add_argument("--year", type=int, required=True, help="Model year")
    p_deal.add_argument("--mileage", type=float, required=True, help="Odometer distance in km")
    p_deal.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 3. depreciation
    p_dep = subparsers.add_parser("depreciation", help="Forecast 5-year future depreciation trajectory")
    p_dep.add_argument("--make", type=str, required=True, help="Manufacturer brand")
    p_dep.add_argument("--model", type=str, required=True, help="Vehicle model name")
    p_dep.add_argument("--year", type=int, required=True, help="Model year")
    p_dep.add_argument("--mileage", type=float, required=True, help="Current odometer distance in km")
    p_dep.add_argument("--annual-mileage", type=float, default=15_000.0, help="Expected driving distance per year in km")
    p_dep.add_argument("--years-ahead", type=int, default=5, help="Forecast horizon in years (1-5)")
    p_dep.add_argument("--json", action="store_true", help="Output trajectory in JSON format")

    # 4. batch
    p_batch = subparsers.add_parser("batch", help="Batch appraise vehicles from a CSV dataset")
    p_batch.add_argument("--input", type=str, required=True, help="Input CSV path")
    p_batch.add_argument("--output", type=str, help="Output path for enriched valuations (.csv or .json)")
    p_batch.add_argument("--limit", type=int, default=50, help="Maximum number of rows to evaluate")
    p_batch.add_argument("--json", action="store_true", help="Output summary in JSON format")

    # 5. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Measure valuation accuracy and latency")
    p_bench.add_argument("--samples", type=int, default=25, help="Number of vehicles to benchmark")
    p_bench.add_argument("--json", action="store_true", help="Output benchmark metrics as JSON")

    return parser


def handle_predict(args: argparse.Namespace) -> int:
    """Execute predict subcommand."""
    t0 = time.time()
    engine = CarValuationEngine()
    explainer = ValuationExplainer()

    val = engine.predict_valuation(
        make=args.make,
        model=args.model,
        year=args.year,
        mileage=args.mileage,
        color=args.color,
        city=args.city,
        automatic_transmission=args.transmission,
    )

    fair_price = val["fair_market_value"]
    quantiles = val["quantiles"]
    explanation = explainer.explain_valuation(
        make=args.make,
        model=args.model,
        year=args.year,
        mileage=args.mileage,
        fair_price=fair_price,
        automatic_transmission=args.transmission,
        city=args.city,
    )

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    payload = {
        "vehicle": val["vehicle"],
        "fair_market_value_egp": fair_price,
        "quantile_bands": quantiles,
        "explanation": explanation,
        "latency_ms": elapsed_ms,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 65)
    print(f"🚗 USED CAR VALUATION REPORT: {args.make} {args.model} ({args.year})")
    print("=" * 65)
    print(f"Fair Market Value:    EGP {fair_price:,.0f}")
    print(f"Confidence Range:     EGP {quantiles['p10']:,.0f} (P10)  ◄─►  EGP {quantiles['p90']:,.0f} (P90)")
    print(f"80% Interval Spread:  EGP {quantiles['interval_80']:,.0f} (Uncertainty: ±{quantiles['uncertainty_pct']}%)")
    print(f"Odometer Reading:     {args.mileage:,.0f} km ({val['vehicle']['mileage_per_year']:,.0f} km/yr)")
    print(f"Inference Latency:    {elapsed_ms} ms")

    if args.verbose:
        print("\n📊 Full Quantile Price Distribution:")
        print(f"  • P10 (Bargain Floor):        EGP {quantiles['p10']:,.0f}")
        print(f"  • P25 (Competitive Entry):    EGP {quantiles['p25']:,.0f}")
        print(f"  • P50 (Market Median):        EGP {quantiles['p50']:,.0f}")
        print(f"  • P75 (Upper Fair Range):     EGP {quantiles['p75']:,.0f}")
        print(f"  • P90 (Dealer Ceiling):       EGP {quantiles['p90']:,.0f}")

        print("\n💡 Key Valuation Drivers:")
        for d in explanation["drivers"]:
            print(f"  • {d['feature']:<25} [{d['impact']}]: {d['detail']}")

    print("\n📝 Market Summary:")
    print(f"  {explanation['summary_rationale']}")
    print("=" * 65)
    return 0


def handle_deal_check(args: argparse.Namespace) -> int:
    """Execute deal-check subcommand."""
    evaluator = DealEvaluator()
    res = evaluator.evaluate_deal(
        asking_price=args.asking_price,
        make=args.make,
        model=args.model,
        year=args.year,
        mileage=args.mileage,
    )

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print("\n" + "=" * 65)
    print(f"🏷️  FAIR MARKET DEAL ADVISOR: {args.make} {args.model} ({args.year})")
    print("=" * 65)
    print(f"Advertised Asking Price: EGP {args.asking_price:,.0f}")
    print(f"Estimated Fair Median:   EGP {res['fair_market_median']:,.0f}")
    print(f"Deal Assessment:         {res['badge']} (Score: {res['deal_score']}/100)")

    if res["is_savings"]:
        print(f"Price Difference:        🟢 EGP {abs(res['price_difference_egp']):,.0f} BELOW market ({abs(res['price_difference_pct']):.1f}% savings)")
    else:
        print(f"Price Difference:        🔴 EGP {abs(res['price_difference_egp']):,.0f} ABOVE market ({abs(res['price_difference_pct']):.1f}% premium)")

    print(f"\nNegotiation Advice:\n  {res['negotiation_advice']}")
    print("=" * 65)
    return 0


def handle_depreciation(args: argparse.Namespace) -> int:
    """Execute depreciation subcommand."""
    forecaster = DepreciationForecaster()
    res = forecaster.forecast_depreciation(
        make=args.make,
        model=args.model,
        year=args.year,
        mileage=args.mileage,
        annual_mileage=args.annual_mileage,
        years_ahead=args.years_ahead,
    )

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    summary = res["summary"]
    print("\n" + "=" * 70)
    print(f"📉 5-YEAR DEPRECIATION TRAJECTORY: {args.make} {args.model} ({args.year})")
    print("=" * 70)
    print(f"Current Valuation:       EGP {res['current_fair_price']:,.0f}")
    print(f"5-Year Residual Value:   {summary['five_year_residual_pct']}% of current value")
    print(f"Total 5-Year Loss:       EGP {summary['five_year_loss_egp']:,.0f} (-{summary['five_year_depreciation_pct']}%)")
    print(f"Average Annual Decay:    {summary['average_annual_depreciation_pct']}% / year")
    print(f"Retention Tier:          {summary['retention_tier']}")
    print("-" * 70)
    print(f"{'Year':<6} {'Calendar':<10} {'Mileage (Km)':<15} {'Estimated Price':<18} {'Residual %':<12}")
    print("-" * 70)
    for pt in res["trajectory"]:
        print(f"+{pt['year_offset']} yr  {pt['calendar_year']:<10} {pt['projected_mileage']:<15,.0f} EGP {pt['estimated_price']:<14,.0f} {pt['residual_value_pct']:>6.1f}%")
    print("=" * 70)
    return 0


def handle_batch(args: argparse.Namespace) -> int:
    """Execute batch subcommand."""
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(in_path)
    evaluator = DealEvaluator()

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    for _, row in df.head(args.limit).iterrows():
        name = str(row.get("Name", ""))
        make = str(row.get("Make", "Other"))
        model = str(row.get("Model", "Other"))
        year = parse_year_from_name(name) or 2020
        mileage = parse_mileage(row.get("Mileage")) or 100_000.0
        asking = parse_price(row.get("Price"))

        val = evaluator.engine.predict_valuation(make=make, model=model, year=year, mileage=mileage)
        fair_price = val["fair_market_value"]

        deal_rating = "Unlisted"
        if asking and asking > 0:
            deal_res = evaluator.evaluate_deal(asking_price=asking, make=make, model=model, year=year, mileage=mileage)
            deal_rating = deal_res["deal_rating"]

        results.append(
            {
                "Make": make,
                "Model": model,
                "Year": year,
                "Mileage": mileage,
                "Asking_Price": asking,
                "Fair_Market_Value": fair_price,
                "P10_Bargain": val["quantiles"]["p10"],
                "P90_Ceiling": val["quantiles"]["p90"],
                "Deal_Rating": deal_rating,
            }
        )

    elapsed = time.time() - t0
    enriched_df = pd.DataFrame(results)

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        else:
            enriched_df.to_csv(out_path, index=False)
        if not args.json:
            print(f"Saved {len(enriched_df)} appraised listings to: {args.output}")

    summary = {
        "processed_vehicles": len(enriched_df),
        "total_time_seconds": round(elapsed, 2),
        "throughput_cars_per_sec": round(len(enriched_df) / max(0.001, elapsed), 2),
        "deal_ratings_breakdown": enriched_df["Deal_Rating"].value_counts().to_dict(),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n" + "=" * 65)
        print("📊 BATCH VEHICLE VALUATION SUMMARY")
        print("=" * 65)
        print(f"Processed:   {summary['processed_vehicles']} vehicles in {summary['total_time_seconds']}s")
        print(f"Throughput:  {summary['throughput_cars_per_sec']} vehicles/sec")
        print("\nDeal Rating Distribution:")
        for rating, count in summary["deal_ratings_breakdown"].items():
            print(f"  • {rating:<20} : {count}")
        print("=" * 65)

    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark subcommand."""
    engine = CarValuationEngine()
    sample_path = Path(__file__).resolve().parent / "data" / "sample_used_cars.csv"
    if not sample_path.exists():
        print("Error: data/sample_used_cars.csv not found.", file=sys.stderr)
        return 1

    df = pd.read_csv(sample_path).head(args.samples)
    latencies = []

    for _, row in df.iterrows():
        make = str(row.get("Make", "Toyota"))
        model = str(row.get("Model", "Corolla"))
        year = parse_year_from_name(row.get("Name")) or 2020
        mileage = parse_mileage(row.get("Mileage")) or 80_000.0

        t0 = time.time()
        _ = engine.predict_valuation(make=make, model=model, year=year, mileage=mileage)
        latencies.append((time.time() - t0) * 1000)

    mean_lat = round(float(sum(latencies) / max(1, len(latencies))), 2)

    payload = {
        "benchmark": "Automotive Valuation & Quantile Uncertainty",
        "evaluated_vehicles": len(df),
        "mean_latency_ms": mean_lat,
        "min_latency_ms": round(float(min(latencies)), 2),
        "max_latency_ms": round(float(max(latencies)), 2),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 60)
        print("⏱️  VALUATION ENGINE LATENCY BENCHMARK")
        print("=" * 60)
        print(f"Evaluated Vehicles:     {payload['evaluated_vehicles']}")
        print(f"Mean Inference Latency: {mean_lat} ms / vehicle")
        print(f"Min / Max Latency:      {payload['min_latency_ms']} ms / {payload['max_latency_ms']} ms")
        print("=" * 60)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "predict":
        return handle_predict(args)
    elif args.subcommand == "deal-check":
        return handle_deal_check(args)
    elif args.subcommand == "depreciation":
        return handle_depreciation(args)
    elif args.subcommand == "batch":
        return handle_batch(args)
    elif args.subcommand == "benchmark":
        return handle_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
