#!/usr/bin/env python
"""Create paper figures from the k=55 Sudoku test-time scaling summary."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


MODELS = ("hrm", "trm", "rt")
TRAIN_BANDS = ("easy", "hard")
TEST_BANDS = ("easy", "hard")
CONDITIONS = {
    "easy_k55_hrm": ("hrm", "easy"),
    "hard_k55_hrm": ("hrm", "hard"),
    "easy_k55_trm": ("trm", "easy"),
    "hard_k55_trm": ("trm", "hard"),
    "easy_k55_rt": ("rt", "easy"),
    "hard_k55_rt": ("rt", "hard"),
}
MODEL_STYLE = {
    "hrm": {"color": "#4285F4", "label": "HRM"},
    "trm": {"color": "#FF5C00", "label": "TRM"},
    "rt": {"color": "#FFC94D", "label": "RT"},
}


@dataclass(frozen=True)
class CurvePoint:
    model: str
    train_band: str
    test_band: str
    l_cycles: int
    mean: float
    seed_sd: float
    num_seeds: int


def load_summary(path: Path) -> list[CurvePoint]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "model", "train_band", "L_cycles", "num_seeds",
        "easy_exact_match_mean", "easy_exact_match_std",
        "hard_exact_match_mean", "hard_exact_match_std",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path} is not a test-time scaling summary CSV")

    points = []
    for row in rows:
        for test_band in TEST_BANDS:
            points.append(CurvePoint(
                model=row["model"],
                train_band=row["train_band"],
                test_band=test_band,
                l_cycles=int(row["L_cycles"]),
                mean=100 * float(row[f"{test_band}_exact_match_mean"]),
                seed_sd=100 * float(row[f"{test_band}_exact_match_std"]),
                num_seeds=int(row["num_seeds"]),
            ))
    validate_points(points)
    return points


def validate_points(points: list[CurvePoint]) -> None:
    expected = {(model, train, test) for model in MODELS for train in TRAIN_BANDS for test in TEST_BANDS}
    grouped: dict[tuple[str, str, str], list[CurvePoint]] = {}
    for point in points:
        key = (point.model, point.train_band, point.test_band)
        grouped.setdefault(key, []).append(point)
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise ValueError(f"summary does not contain the complete 3 x 2 x 2 matrix; missing={missing}, extra={extra}")
    l_values = {point.l_cycles for point in next(iter(grouped.values()))}
    for key, group in grouped.items():
        if {point.l_cycles for point in group} != l_values:
            raise ValueError(f"inconsistent L values for {key}")
        if any(point.num_seeds < 2 for point in group):
            raise ValueError(f"{key} contains fewer than two seeds")


def parse_exclusion(value: str) -> tuple[str, int]:
    condition, separator, seed_text = value.partition(":")
    if separator != ":" or condition not in CONDITIONS:
        raise argparse.ArgumentTypeError("--exclude must be CONDITION:SEED for a k=55 condition")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--exclude seed must be an integer") from error
    if seed < 0:
        raise argparse.ArgumentTypeError("--exclude seed must be non-negative")
    return condition, seed


def points_from_workers(workers_dir: Path, exclusions: set[tuple[str, int]]) -> list[CurvePoint]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    seen: set[tuple[str, int, int, str]] = set()
    worker_files = sorted(workers_dir.glob("*/test_time_scaling_per_seed.csv"))
    if not worker_files:
        raise FileNotFoundError(f"No per-seed worker results found under {workers_dir}")
    for path in worker_files:
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            condition = row["condition"]
            seed = int(row["seed"])
            if (condition, seed) in exclusions:
                continue
            if condition not in CONDITIONS:
                raise ValueError(f"Unexpected condition in {path}: {condition}")
            key = (condition, seed, int(row["L_cycles"]), row["test_band"])
            if key in seen:
                raise ValueError(f"Duplicate worker result for {key}")
            seen.add(key)
            model, train_band = CONDITIONS[condition]
            grouped.setdefault((model, train_band, row["test_band"], int(row["L_cycles"])), []).append(
                100 * float(row["exact_match_accuracy"])
            )
    points = [
        CurvePoint(
            model=model,
            train_band=train_band,
            test_band=test_band,
            l_cycles=l_cycles,
            mean=float(np.mean(values)),
            seed_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            num_seeds=len(values),
        )
        for (model, train_band, test_band, l_cycles), values in grouped.items()
    ]
    validate_points(points)
    return points


def write_summary(path: Path, points: list[CurvePoint]) -> None:
    rows: dict[tuple[str, str, int], dict[str, float | int | str]] = {}
    for point in points:
        row = rows.setdefault(
            (point.model, point.train_band, point.l_cycles),
            {
                "model": point.model,
                "train_band": point.train_band,
                "L_cycles": point.l_cycles,
                "num_seeds": point.num_seeds,
            },
        )
        row[f"{point.test_band}_exact_match_mean"] = point.mean / 100
        row[f"{point.test_band}_exact_match_std"] = point.seed_sd / 100
    fields = [
        "model", "train_band", "L_cycles", "num_seeds",
        "easy_exact_match_mean", "easy_exact_match_std",
        "hard_exact_match_mean", "hard_exact_match_std",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (str(row["model"]), str(row["train_band"]), int(row["L_cycles"]))))


def panel_points(points: list[CurvePoint], train_band: str, test_band: str, model: str) -> list[CurvePoint]:
    return sorted(
        [
            point for point in points
            if point.train_band == train_band and point.test_band == test_band and point.model == model
        ],
        key=lambda point: point.l_cycles,
    )


def draw_panel(axis: plt.Axes, points: list[CurvePoint], train_band: str, test_band: str) -> None:
    for model in MODELS:
        curve = panel_points(points, train_band, test_band, model)
        x = np.asarray([point.l_cycles for point in curve])
        mean = np.asarray([point.mean for point in curve])
        seed_sd = np.asarray([point.seed_sd for point in curve])
        style = MODEL_STYLE[model]
        axis.fill_between(x, mean - seed_sd, mean + seed_sd, color=style["color"], alpha=0.16, linewidth=0)
        axis.plot(x, mean, color=style["color"], marker="o", markersize=4, linewidth=2, label=style["label"])
    axis.axvline(6, color="#4D4D4D", linestyle="--", linewidth=1, zorder=0)
    axis.set_xscale("log", base=2)
    axis.set_xticks([6, 8, 16, 32, 64, 128, 256])
    axis.set_xticklabels(["6", "8", "16", "32", "64", "128", "256"])
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_title(f"{train_band.capitalize()} train / {test_band.capitalize()} test", fontsize=10)


def shared_legend() -> list[Line2D]:
    handles = [
        Line2D([], [], color=MODEL_STYLE[model]["color"], marker="o", linewidth=2, label=MODEL_STYLE[model]["label"])
        for model in MODELS
    ]
    handles.append(Line2D([], [], color="#4D4D4D", linestyle="--", linewidth=1, label="Training L = 6"))
    return handles


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_hard_test(points: list[CurvePoint], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for axis, train_band in zip(axes, TRAIN_BANDS):
        draw_panel(axis, points, train_band, "hard")
        axis.set_xlabel("Inference L cycles")
    axes[0].set_ylabel("Hard-test exact match (%)")
    axes[0].set_ylim(20, 100)
    figure.legend(handles=shared_legend(), ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    figure.tight_layout()
    save_figure(figure, output_dir, "k55_test_time_scaling_hard_test")


def plot_all_tests(points: list[CurvePoint], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), sharex=True, sharey=True)
    for row, train_band in enumerate(TRAIN_BANDS):
        for column, test_band in enumerate(TEST_BANDS):
            axis = axes[row, column]
            draw_panel(axis, points, train_band, test_band)
            if row == len(TRAIN_BANDS) - 1:
                axis.set_xlabel("Inference L cycles")
            if column == 0:
                axis.set_ylabel("Exact match (%)")
    axes[0, 0].set_ylim(20, 100)
    figure.legend(handles=shared_legend(), ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    figure.tight_layout()
    save_figure(figure, output_dir, "k55_test_time_scaling_all_tests")


def l6_table_rows(points: list[CurvePoint]) -> list[list[str]]:
    rows = []
    for train_band in TRAIN_BANDS:
        for test_band in TEST_BANDS:
            values = []
            for model in MODELS:
                matching = [
                    point for point in points
                    if point.model == model
                    and point.train_band == train_band
                    and point.test_band == test_band
                    and point.l_cycles == 6
                ]
                if len(matching) != 1:
                    raise ValueError(f"expected one L=6 result for {model}/{train_band}/{test_band}")
                point = matching[0]
                values.append(f"{point.mean:.2f} +/- {point.seed_sd:.2f}")
            rows.append([f"{train_band.capitalize()} train", f"{test_band.capitalize()} test", *values])
    return rows


def plot_l6_table(points: list[CurvePoint], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 2.55))
    axis.axis("off")
    table = axis.table(
        cellText=l6_table_rows(points),
        colLabels=["Training data", "Test data", "HRM", "TRM", "RT"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.19, 0.16, 0.215, 0.215, 0.215],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    header_colors = ["#4D4D4D", "#4D4D4D", *[MODEL_STYLE[model]["color"] for model in MODELS]]
    for column, color in enumerate(header_colors):
        cell = table[(0, column)]
        cell.set_facecolor(color)
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")
    for row in range(1, 5):
        row_color = "#F3F3F3" if row in {1, 2} else "#FFFFFF"
        for column in range(5):
            cell = table[(row, column)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor("#D0D0D0")
            if column < 2:
                cell.set_text_props(weight="bold" if column == 0 else "normal")
    axis.set_title("L = 6 (training-time inference budget); exact match %, mean +/- seed SD", pad=12, fontsize=10)
    save_figure(figure, output_dir, "k55_l6_accuracy_table")


def l6_bar_data(points: list[CurvePoint]) -> tuple[list[str], dict[str, list[CurvePoint]]]:
    labels = ["Easy -> Easy", "Easy -> Hard", "Hard -> Easy", "Hard -> Hard"]
    grouped = {
        model: [
            next(
                point for point in points
                if point.model == model
                and point.train_band == train_band
                and point.test_band == test_band
                and point.l_cycles == 6
            )
            for train_band in TRAIN_BANDS for test_band in TEST_BANDS
        ]
        for model in MODELS
    }
    return labels, grouped


def plot_l6_bars(points: list[CurvePoint], output_dir: Path) -> None:
    labels, grouped = l6_bar_data(points)
    figure, axis = plt.subplots(figsize=(7.2, 3.65))
    x = np.arange(len(labels), dtype=float)
    width = 0.23
    for index, model in enumerate(MODELS):
        curve = grouped[model]
        values = np.asarray([point.mean for point in curve])
        errors = np.asarray([point.seed_sd for point in curve])
        positions = x + (index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            color=MODEL_STYLE[model]["color"],
            edgecolor="#333333",
            linewidth=0.55,
            label=MODEL_STYLE[model]["label"],
            yerr=errors,
            capsize=3,
            error_kw={"ecolor": "#333333", "elinewidth": 0.9, "capthick": 0.9},
        )
        for bar, value, error in zip(bars, values, errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + 1.25,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Exact-match accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    axis.set_title("L = 6 (training-time inference budget); bars show mean, error bars show seed SD", fontsize=10, pad=16)
    figure.tight_layout()
    save_figure(figure, output_dir, "k55_l6_accuracy_bars")


def write_paper_notes(output_dir: Path, exclusions: set[tuple[str, int]]) -> None:
    sensitivity_note = ""
    if exclusions:
        excluded = ", ".join(f"{condition}/seed_{seed}" for condition, seed in sorted(exclusions))
        sensitivity_note = (
            f"\nSensitivity-analysis note: this directory excludes `{excluded}`. "
            "It is useful for diagnosing the curve, but the original three-seed analysis should remain primary.\n"
        )
    (output_dir / "k55_test_time_scaling_caption.md").write_text(
        "Suggested main-text caption:\n\n"
        "**Test-time scaling on Sudoku with the k=55 split.** Models are trained on Easy "
        "(at most 55 blanks) or Hard (at least 56 blanks) puzzles with H=2, L=6, "
        "then evaluated without parameter updates at larger inference-time L budgets. "
        "Lines show the mean exact-match accuracy across three training seeds; shaded "
        "bands show one standard deviation. The dashed line marks the training-time "
        "budget. TRM continues to benefit from additional inference computation, RT "
        "improves smoothly, while HRM becomes unstable beyond short rollouts.\n\n"
        "Reporting note: these results use the currently selected best epoch for each seed. "
        "If `eval/hard_exact_match` is described as a test metric, do not present this as "
        "a strictly test-only model-selection protocol; use a separate validation split or "
        "select a fixed epoch before reporting final test results.\n"
        + sensitivity_note
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv", type=Path,
        default=Path("results/k55_test_time_scaling/test_time_scaling_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/k55_test_time_scaling/figures"),
    )
    parser.add_argument(
        "--workers-dir", type=Path,
        help="Worker result directory used when recomputing a sensitivity analysis.",
    )
    parser.add_argument(
        "--exclude", action="append", type=parse_exclusion, default=[], metavar="CONDITION:SEED",
        help="Exclude one condition/seed and recompute the plotted means and seed standard deviations.",
    )
    args = parser.parse_args()

    exclusions = set(args.exclude)
    if exclusions:
        workers_dir = args.workers_dir or args.summary_csv.parent / "workers"
        points = points_from_workers(workers_dir, exclusions)
    else:
        points = load_summary(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if exclusions:
        write_summary(args.output_dir / "test_time_scaling_summary.csv", points)
    plot_hard_test(points, args.output_dir)
    plot_all_tests(points, args.output_dir)
    plot_l6_table(points, args.output_dir)
    plot_l6_bars(points, args.output_dir)
    write_paper_notes(args.output_dir, exclusions)
    print(f"Wrote paper figures to {args.output_dir}")


if __name__ == "__main__":
    main()
