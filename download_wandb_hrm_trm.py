"""Download and compare five-seed HRM/TRM held-out evaluation results.

For each model x training-band x test-band cell, ``best_accuracy`` is the
largest logged exact-match accuracy over all completed five-seed runs and all
their epochs. The winning seed and W&B step are retained for traceability.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import stdev
from typing import Any

import wandb


MODELS = ("hrm", "trm")
BANDS = ("easy", "medium", "hard")
METRICS = {band: f"eval/{band}_exact_match" for band in BANDS}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history_records(run: Any) -> list[dict[str, Any]]:
    keys = ["_step", *METRICS.values()]
    # These runs log exactly one complete evaluation row per epoch. Requesting
    # the keyed history avoids scanning tens of thousands of train-only steps.
    return run.history(samples=1_000, keys=keys).to_dict("records")


def _best_metrics(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, tuple[float, int]] = {}
    for row in history:
        step = int(row.get("_step", -1))
        for test_band, metric in METRICS.items():
            value = row.get(metric)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            previous = best.get(test_band)
            if previous is None or value > previous[0] or (value == previous[0] and step < previous[1]):
                best[test_band] = (float(value), step)

    return [
        {
            "test_band": test_band,
            "best_accuracy": accuracy,
            "best_step": step,
        }
        for test_band, (accuracy, step) in best.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="classicalqy-peking-university")
    parser.add_argument("--project", default="hrm-trm")
    parser.add_argument("--output-dir", default="results/wandb_hrm_trm_five_seed")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--train-bands", nargs="+", choices=BANDS, default=list(BANDS))
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    expected_groups = {
        f"{train_band}_{model}_five_seed": (model, train_band)
        for model in args.models for train_band in args.train_bands
    }
    rows: list[dict[str, Any]] = []
    curve_values: dict[tuple[str, str, str, int], list[float]] = {}
    for run in api.runs(f"{args.entity}/{args.project}", per_page=100):
        group = run.group or ""
        if group not in expected_groups or run.state != "finished":
            continue
        model, train_band = expected_groups[group]
        seed = run.config.get("seed")
        history = _history_records(run)
        for row in history:
            step = int(row.get("_step", -1))
            for test_band, metric_name in METRICS.items():
                value = row.get(metric_name)
                if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                curve_values.setdefault((model, train_band, test_band, step), []).append(float(value))
        for metric in _best_metrics(history):
            rows.append({
                "model": model,
                "train_band": train_band,
                "seed": seed,
                "run_id": run.id,
                "run_name": run.name,
                **metric,
            })

    if not rows:
        raise RuntimeError("no finished five-seed HRM/TRM runs with evaluation metrics were found")

    rows.sort(key=lambda row: (row["model"], row["train_band"], str(row["seed"]), row["test_band"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_run_best.csv", rows)

    winners: list[dict[str, Any]] = []
    for model in MODELS:
        for train_band in BANDS:
            for test_band in BANDS:
                candidates = [
                    row for row in rows
                    if row["model"] == model and row["train_band"] == train_band and row["test_band"] == test_band
                ]
                if not candidates:
                    continue
                winner = max(candidates, key=lambda row: (row["best_accuracy"], -row["best_step"]))
                winners.append({
                    "model": model,
                    "train_band": train_band,
                    "test_band": test_band,
                    "best_accuracy": winner["best_accuracy"],
                    "winning_seed": winner["seed"],
                    "winning_step": winner["best_step"],
                    "winning_run_id": winner["run_id"],
                    "completed_seed_count": len({row["seed"] for row in candidates}),
                })

    _write_csv(output_dir / "best_by_cell.csv", winners)

    mean_curve_rows: list[dict[str, Any]] = []
    for model in args.models:
        for train_band in args.train_bands:
            for test_band in BANDS:
                curve = [
                    (step, sum(values) / len(values), len(values))
                    for (curve_model, curve_train, curve_test, step), values in curve_values.items()
                    if (curve_model, curve_train, curve_test) == (model, train_band, test_band)
                ]
                if not curve:
                    continue
                best_step, best_mean, seed_count = max(curve, key=lambda item: (item[1], -item[0]))
                best_values = curve_values[(model, train_band, test_band, best_step)]
                mean_curve_rows.append({
                    "model": model,
                    "train_band": train_band,
                    "test_band": test_band,
                    "best_mean_accuracy": best_mean,
                    "std_accuracy": stdev(best_values) if len(best_values) > 1 else 0.0,
                    "best_step": best_step,
                    "seed_count_at_best_step": seed_count,
                })
    _write_csv(output_dir / "best_mean_curve_by_cell.csv", mean_curve_rows)
    matrices: dict[str, dict[str, dict[str, dict[str, Any]]]] = {model: {} for model in args.models}
    for winner in winners:
        matrices[winner["model"]].setdefault(winner["train_band"], {})[winner["test_band"]] = winner
    with (output_dir / "best_matrices.json").open("w") as file:
        json.dump(matrices, file, indent=2)

    comparison = []
    indexed = {(row["model"], row["train_band"], row["test_band"]): row for row in winners}
    for train_band in args.train_bands:
        for test_band in BANDS:
            hrm = indexed.get(("hrm", train_band, test_band))
            trm = indexed.get(("trm", train_band, test_band))
            if hrm is None or trm is None:
                continue
            comparison.append({
                "train_band": train_band,
                "test_band": test_band,
                "hrm_best_accuracy": hrm["best_accuracy"],
                "trm_best_accuracy": trm["best_accuracy"],
                "trm_minus_hrm": trm["best_accuracy"] - hrm["best_accuracy"],
                "hrm_seed": hrm["winning_seed"],
                "trm_seed": trm["winning_seed"],
            })
    if comparison:
        _write_csv(output_dir / "hrm_vs_trm_best_comparison.csv", comparison)
    print(f"Saved {len(rows)} per-run metrics and {len(winners)} best cells to {output_dir}")


if __name__ == "__main__":
    main()
