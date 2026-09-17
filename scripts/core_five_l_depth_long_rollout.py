#!/usr/bin/env python
"""Physical-time L-depth sweep for the core-five deterministic MSD analysis.

Unlike the original core-five profile, this collector samples HRM after every
underlying L update.  It therefore compares inference-time L schedules on one
common physical clock, including schedules with very few H updates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import filecmp
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from arch.layers import Carry
from scripts.analyze_long_rollout_msd import RunDirectory, build_model, data_kwargs, load_module
from scripts.core_five_long_rollout import (
    CORE_CONDITIONS,
    core_runs,
    fixed_random_samples,
    read_csv,
)
from scripts.long_rollout_msd_utils import STATE_NAMES, log_spaced_lags, segment_for_pair, state_msd_per_puzzle, rt_state_msd_per_puzzle


H2L6_CONDITIONS = ("H2L6_h", "H2L6_l", "H2L6_hl")
CONTROL_CONDITIONS = ("H2L1_h", "RT")
DEFAULT_EVAL_L_VALUES = (6, 8, 16, 32, 64, 128, 256, 512, 1024)
# Canonical inference-depth palette.  The same depth keeps the same colour in
# all L-depth and H/L-clock figures, independent of readout or time window.
L_DEPTH_COLORS = {
    6: "#5E3C99", 8: "#3F60AA", 16: "#2C7FB8", 32: "#1B9E77",
    64: "#66A61E", 128: "#E6AB02", 256: "#E76F51", 512: "#D73027",
    1024: "#8C510A",
}
PHYSICAL_BOUNDARIES = np.asarray((0, 48, 192, 768, 4096), dtype=np.int64)
SCHEME = "core_five_l_depth_physical_l_updates_v1"
MIN_OUTER_SCHEME = "core_five_l_depth_min_outer16_v1"
SELECTION_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "epoch", "checkpoint",
    "test_exact_match", "cell_accuracy", "evaluated_examples", "selection_policy",
    "selection_completed",
]
METADATA_FIELDS = [
    "kind", "condition", "readout", "train_l", "eval_l", "seed", "best_epoch", "checkpoint", "samples",
    "sample_seed", "sample_manifest_sha256", "requested_l_updates", "actual_l_updates", "completed_h_updates",
    "tail_l_updates", "segment_boundaries_l_updates", "lag_l_updates", "state_shape", "per_puzzle_file",
]
SUMMARY_FIELDS = [
    "kind", "condition", "readout", "train_l", "eval_l", "seed", "state", "segment", "lag_l_updates",
    "mean", "median", "ci95_low", "ci95_high", "puzzles", "origins", "status",
]
SEED_FIELDS = [
    "kind", "condition", "readout", "train_l", "eval_l", "state", "segment", "lag_l_updates",
    "seed_mean", "cluster_ci95_low", "cluster_ci95_high", "seeds", "puzzles_per_seed", "status",
]
RATIO_FIELDS = [
    "condition", "readout", "eval_l", "seed", "segment", "lag_l_updates",
    "msd_h", "msd_l", "h_over_l", "status",
]


@dataclass(frozen=True)
class SweepUnit:
    run: RunDirectory
    eval_l: int

    @property
    def key(self) -> tuple[str, int, int]:
        return self.run.condition, self.run.seed, self.eval_l

    @property
    def label(self) -> str:
        return f"{self.run.condition}/L{self.eval_l}/seed_{self.run.seed}"


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def absolute_checkpoint(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def lags_for_boundaries(boundaries: np.ndarray, points: int) -> np.ndarray:
    lags: set[int] = set()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        lags.update(log_spaced_lags(int(end - start - 1), points))
    return np.asarray(sorted(lags), dtype=np.int64)


def rollout_spec(unit: SweepUnit, args: Any) -> tuple[int, np.ndarray, str]:
    """Return physical horizon and four segment endpoints for this rollout unit."""
    if not getattr(args, "min_outer_cycles", None):
        return int(PHYSICAL_BOUNDARIES[-1]), PHYSICAL_BOUNDARIES, SCHEME
    minimum_outer = int(args.min_outer_cycles)
    if unit.run.kind == "rt":
        updates = int(PHYSICAL_BOUNDARIES[-1])
        return updates, np.linspace(0, updates, 5, dtype=np.int64), MIN_OUTER_SCHEME
    # One normal H=2 outer call performs 2 * eval_l lower updates.  The horizon
    # is a multiple of this span whenever the min-outer constraint is active.
    updates = max(int(PHYSICAL_BOUNDARIES[-1]), 2 * minimum_outer * unit.eval_l)
    completed_outer, _tail = divmod(updates, 2 * unit.eval_l)
    outer_boundaries = np.asarray([0, completed_outer // 4, completed_outer // 2,
                                   3 * completed_outer // 4, completed_outer], dtype=np.int64)
    lower_boundaries = outer_boundaries * (2 * unit.eval_l)
    lower_boundaries[-1] = updates  # Keep an incomplete final outer-call tail in the last segment.
    return updates, lower_boundaries, MIN_OUTER_SCHEME


def expected_h_updates(eval_l: int, updates: int = 4096) -> tuple[int, int]:
    if eval_l < 1:
        raise ValueError("eval_l must be positive")
    return divmod(updates, eval_l)


def units_from_runs(runs: list[RunDirectory], seeds: tuple[int, ...], l_values: tuple[int, ...]) -> list[SweepUnit]:
    lookup = {(run.condition, run.seed): run for run in runs}
    missing = [f"{condition}/seed_{seed}" for condition in CORE_CONDITIONS for seed in seeds if (condition, seed) not in lookup]
    if missing:
        raise FileNotFoundError("Missing core-five runs:\n" + "\n".join(missing))
    units = [SweepUnit(lookup[(condition, seed)], eval_l)
             for condition in H2L6_CONDITIONS for eval_l in l_values for seed in seeds]
    units.extend(SweepUnit(lookup[(condition, seed)], 1) for condition in CONTROL_CONDITIONS for seed in seeds)
    return units


def selected_checkpoints(path: Path, units: list[SweepUnit]) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(path)
    lookup = {(row["condition"], int(row["seed"])): row for row in rows}
    expected = {(unit.run.condition, unit.run.seed) for unit in units}
    missing = sorted(expected - set(lookup))
    if missing:
        raise FileNotFoundError(f"Reference best checkpoints are incomplete in {path}: {missing}")
    for key in expected:
        checkpoint = absolute_checkpoint(lookup[key]["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Reference checkpoint is missing: {checkpoint}")
    return {key: lookup[key] for key in expected}


def write_manifest(output_dir: Path, indices: np.ndarray, sample_seed: int) -> str:
    rows = [{"sample_position": index, "test_hard_stream_index": int(value), "sample_seed": sample_seed}
            for index, value in enumerate(indices)]
    atomic_csv(output_dir / "sample_manifest.csv", rows, ["sample_position", "test_hard_stream_index", "sample_seed"])
    digest = hashlib.sha256(",".join(str(value) for value in indices).encode()).hexdigest()
    (output_dir / "sample_manifest.sha256").write_text(digest + "\n")
    return digest


def initial_hrm_state(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, l = model.initial_carry["z_H"], model.initial_carry["z_L"]  # type: ignore[attr-defined]
    shape = (input_ids.shape[0], input_ids.shape[1], h.shape[-1])
    if h.ndim < 3:
        h = h.reshape(1, 1, -1).expand(shape)
    if l.ndim < 3:
        l = l.reshape(1, 1, -1).expand(shape)
    return h, l


def initial_rt_state(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    z = model.initial_carry["z"]  # type: ignore[attr-defined]
    if z.ndim < 3:
        z = z.reshape(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
    return z


@torch.inference_mode()
def advance_hrm_l(model: torch.nn.Module, state: tuple[torch.Tensor, torch.Tensor], embedding: torch.Tensor,
                  phase: int, eval_l: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int, bool]:
    """Advance exactly one L update and conditionally perform the H update."""
    h, l = state
    lower = model.L_level if hasattr(model, "L_level") else model.core  # type: ignore[attr-defined]
    upper = model.H_level if hasattr(model, "H_level") else model.core  # type: ignore[attr-defined]
    l = lower(l + h + embedding)
    phase += 1
    updated_h = phase == eval_l
    if updated_h:
        h = upper(h + l)
        phase = 0
    return (h, l), phase, updated_h


@torch.inference_mode()
def advance_rt_l(model: torch.nn.Module, state: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    return model.core(state + embedding)  # type: ignore[attr-defined]


def collect_hrm_per_puzzle(model: torch.nn.Module, input_ids: torch.Tensor, eval_l: int, lags: np.ndarray,
                            updates: int, boundaries: np.ndarray, progress: tqdm[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Exact paired-rollout MSD sampled after every underlying L update."""
    values = np.full((input_ids.shape[0], len(STATE_NAMES), 4, len(lags)), np.nan, dtype=np.float32)
    origins = np.zeros((4, len(lags)), dtype=np.int32)
    embedding = model.embed(input_ids)  # type: ignore[attr-defined]
    for lag_index, lag in enumerate(lags.tolist()):
        ref_state = initial_hrm_state(model, input_ids)
        lead_state = initial_hrm_state(model, input_ids)
        ref_phase = lead_phase = 0
        for _ in range(lag):
            lead_state, lead_phase, _ = advance_hrm_l(model, lead_state, embedding, lead_phase, eval_l)
        totals = {segment: {state: torch.zeros(input_ids.shape[0], device=input_ids.device) for state in STATE_NAMES} for segment in range(4)}
        counts = np.zeros(4, dtype=np.int32)
        for time_index in range(updates - lag + 1):
            segment = segment_for_pair(time_index, lag, tuple(boundaries.tolist()))
            if segment is not None:
                for state_name, result in state_msd_per_puzzle(*ref_state, *lead_state).items():
                    totals[segment][state_name] += result
                counts[segment] += 1
            if time_index < updates - lag:
                ref_state, ref_phase, _ = advance_hrm_l(model, ref_state, embedding, ref_phase, eval_l)
                lead_state, lead_phase, _ = advance_hrm_l(model, lead_state, embedding, lead_phase, eval_l)
        for segment in range(4):
            if counts[segment]:
                origins[segment, lag_index] = counts[segment]
                for state_index, state_name in enumerate(STATE_NAMES):
                    values[:, state_index, segment, lag_index] = (totals[segment][state_name] / counts[segment]).cpu().numpy()
                values[:, 3, segment, lag_index] = (values[:, 0, segment, lag_index] + values[:, 1, segment, lag_index]) / 2
        progress.update(1)
    return values, origins


def collect_rt_per_puzzle(model: torch.nn.Module, input_ids: torch.Tensor, lags: np.ndarray, updates: int,
                           boundaries: np.ndarray, progress: tqdm[Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((input_ids.shape[0], 1, 4, len(lags)), np.nan, dtype=np.float32)
    origins = np.zeros((4, len(lags)), dtype=np.int32)
    embedding = model.embed(input_ids)  # type: ignore[attr-defined]
    for lag_index, lag in enumerate(lags.tolist()):
        ref_state = initial_rt_state(model, input_ids)
        lead_state = initial_rt_state(model, input_ids)
        for _ in range(lag):
            lead_state = advance_rt_l(model, lead_state, embedding)
        totals = {segment: torch.zeros(input_ids.shape[0], device=input_ids.device) for segment in range(4)}
        counts = np.zeros(4, dtype=np.int32)
        for time_index in range(updates - lag + 1):
            segment = segment_for_pair(time_index, lag, tuple(boundaries.tolist()))
            if segment is not None:
                totals[segment] += rt_state_msd_per_puzzle(ref_state, lead_state)
                counts[segment] += 1
            if time_index < updates - lag:
                ref_state = advance_rt_l(model, ref_state, embedding)
                lead_state = advance_rt_l(model, lead_state, embedding)
        for segment in range(4):
            if counts[segment]:
                origins[segment, lag_index] = counts[segment]
                values[:, 0, segment, lag_index] = (totals[segment] / counts[segment]).cpu().numpy()
        progress.update(1)
    return values, origins


def unit_filename(unit: SweepUnit) -> str:
    return f"{unit.run.condition}_evalL{unit.eval_l}_seed_{unit.run.seed}.npz"


def metadata_row(unit: SweepUnit, selected: dict[str, str], output_path: Path, samples: int, sample_seed: int,
                 manifest: str, lags: np.ndarray, updates: int, boundaries: np.ndarray) -> dict[str, object]:
    completed_h, tail = expected_h_updates(unit.eval_l, updates) if unit.run.kind in ("hrm", "trm") else (updates, 0)
    return {
        "kind": unit.run.kind, "condition": unit.run.condition, "readout": unit.run.readout,
        "train_l": "" if unit.run.l_cycles is None else unit.run.l_cycles, "eval_l": unit.eval_l,
        "seed": unit.run.seed, "best_epoch": int(selected["epoch"]), "checkpoint": selected["checkpoint"], "samples": samples,
        "sample_seed": sample_seed, "sample_manifest_sha256": manifest, "requested_l_updates": updates,
        "actual_l_updates": updates, "completed_h_updates": completed_h, "tail_l_updates": tail,
        "segment_boundaries_l_updates": json.dumps(boundaries.tolist()), "lag_l_updates": json.dumps(lags.tolist()),
        "state_shape": f"{samples}x82x512", "per_puzzle_file": str(output_path),
    }


def collect_unit(unit: SweepUnit, selected: dict[str, str], fixed_x: torch.Tensor, metadata: dict[str, Any], args: Any,
                 manifest: str, lags: np.ndarray, updates: int, boundaries: np.ndarray, scheme: str,
                 progress: tqdm[Any]) -> dict[str, object]:
    output_path = args.output_dir / "per_puzzle_msd" / unit_filename(unit)
    checkpoint = absolute_checkpoint(selected["checkpoint"])
    if output_path.is_file():
        cached = np.load(output_path, allow_pickle=False)
        if ("analysis_scheme" in cached.files and str(cached["analysis_scheme"].item()) == scheme
                and int(cached["eval_l"].item()) == unit.eval_l
                and np.array_equal(cached["lag_l_updates"], lags)
                and np.array_equal(cached["segment_boundaries_l_updates"], boundaries)
                and "checkpoint" in cached.files and str(cached["checkpoint"].item()) == str(checkpoint)
                and "sample_manifest_sha256" in cached.files
                and str(cached["sample_manifest_sha256"].item()) == manifest):
            return metadata_row(unit, selected, output_path, len(fixed_x), args.sample_seed, manifest, lags, updates, boundaries)
    model = build_model(unit.run, checkpoint, metadata, args.device)
    batches: list[np.ndarray] = []
    origin_reference: np.ndarray | None = None
    for start in range(0, len(fixed_x), args.rollout_batch_size):
        x = fixed_x[start:start + args.rollout_batch_size].to(args.device, non_blocking=True)
        if unit.run.kind in ("hrm", "trm"):
            values, origins = collect_hrm_per_puzzle(model, x, unit.eval_l, lags, updates, boundaries, progress)
        else:
            values, origins = collect_rt_per_puzzle(model, x, lags, updates, boundaries, progress)
        batches.append(values)
        if origin_reference is None:
            origin_reference = origins
        elif not np.array_equal(origin_reference, origins):
            raise AssertionError("Rollout batches produced inconsistent origin counts.")
    assert origin_reference is not None
    all_values = np.concatenate(batches, axis=0)
    if unit.run.kind in ("hrm", "trm"):
        if not np.array_equal(all_values[:, 3], (all_values[:, 0] + all_values[:, 1]) / 2, equal_nan=True):
            raise AssertionError("[H,L] MSD identity failed.")
        states = np.asarray(STATE_NAMES)
    else:
        states = np.asarray(("rt",))
    atomic_npz(output_path, msd=all_values, state_names=states, lag_l_updates=lags,
               segment_boundaries_l_updates=boundaries, origins=origin_reference,
               analysis_scheme=np.asarray(scheme), eval_l=np.asarray(unit.eval_l), checkpoint=np.asarray(str(checkpoint)),
               sample_manifest_sha256=np.asarray(manifest))
    del model; gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata_row(unit, selected, output_path, len(fixed_x), args.sample_seed, manifest, lags, updates, boundaries)


def bootstrap(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, float]:
    mean, median = float(values.mean()), float(np.median(values))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, (.025, .975))
    return mean, median, float(low), float(high)


def summarize(output_dir: Path, repeats: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    puzzle_rows: list[dict[str, object]] = []
    raw: dict[tuple[str, int, int], tuple[dict[str, str], Any]] = {}
    for meta in read_csv(output_dir / "rollout_metadata.csv"):
        arrays = np.load(meta["per_puzzle_file"], allow_pickle=False)
        values = arrays["msd"].astype(np.float64)
        raw[(meta["condition"], int(meta["eval_l"]), int(meta["seed"]))] = (meta, arrays)
        indices = rng.integers(0, values.shape[0], size=(repeats, values.shape[0]))
        for state_index, state in enumerate(arrays["state_names"]):
            for segment in range(4):
                for point, lag in enumerate(arrays["lag_l_updates"]):
                    current = values[:, state_index, segment, point]
                    if not np.isfinite(current).all():
                        continue
                    mean, median, low, high = bootstrap(current, indices)
                    puzzle_rows.append({**{key: meta[key] for key in ("kind", "condition", "readout", "train_l", "eval_l", "seed")},
                                        "state": str(state), "segment": segment + 1, "lag_l_updates": int(lag),
                                        "mean": mean, "median": median, "ci95_low": low, "ci95_high": high,
                                        "puzzles": len(current), "origins": int(arrays["origins"][segment, point]),
                                        "status": "no H update" if str(state) == "h" and np.all(current == 0) else "defined"})
    seed_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int, str], list[tuple[dict[str, str], Any]]] = {}
    for (condition, eval_l, _seed), item in raw.items():
        grouped.setdefault((condition, eval_l, item[0]["readout"]), []).append(item)
    for (_condition, _eval_l, _readout), items in grouped.items():
        items.sort(key=lambda item: int(item[0]["seed"]))
        meta, first = items[0]
        for state_index, state in enumerate(first["state_names"]):
            for segment in range(4):
                for point, lag in enumerate(first["lag_l_updates"]):
                    seed_values = [arrays["msd"][:, state_index, segment, point].astype(np.float64) for _, arrays in items]
                    if not all(np.isfinite(value).all() for value in seed_values):
                        continue
                    seed_means = np.asarray([value.mean() for value in seed_values])
                    draws = np.empty(repeats)
                    for repeat in range(repeats):
                        selected = rng.integers(0, len(seed_values), size=len(seed_values))
                        draws[repeat] = np.mean([seed_values[index][rng.integers(0, len(seed_values[index]), len(seed_values[index]))].mean() for index in selected])
                    seed_rows.append({**{key: meta[key] for key in ("kind", "condition", "readout", "train_l", "eval_l")},
                                      "state": str(state), "segment": segment + 1, "lag_l_updates": int(lag),
                                      "seed_mean": float(seed_means.mean()), "cluster_ci95_low": float(np.quantile(draws, .025)),
                                      "cluster_ci95_high": float(np.quantile(draws, .975)), "seeds": len(seed_values),
                                      "puzzles_per_seed": len(seed_values[0]),
                                      "status": "no H update" if str(state) == "h" and all(np.all(value == 0) for value in seed_values) else "defined"})
    return puzzle_rows, seed_rows


def curve(metadata: list[dict[str, str]], condition: str, eval_l: int, state: str, segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    selected = sorted((row for row in metadata if row["condition"] == condition and int(row["eval_l"]) == eval_l), key=lambda row: int(row["seed"]))
    arrays = [np.load(row["per_puzzle_file"], allow_pickle=False) for row in selected]
    index = [str(value) for value in arrays[0]["state_names"]].index(state)
    values = [array["msd"][:, index, segment, :].astype(np.float64) for array in arrays]
    valid = np.logical_and.reduce([np.isfinite(value).all(axis=0) for value in values])
    lags = arrays[0]["lag_l_updates"][valid].astype(float)
    means = [value[:, valid].mean(axis=0) for value in values]
    average = np.mean(means, axis=0)
    low, high = np.quantile(np.asarray(means), (.025, .975), axis=0)
    return lags, average, low, high, means


def cluster_bounds(rows: list[dict[str, str]], condition: str, eval_l: int, state: str, segment: int,
                   lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["condition"] == condition and int(row["eval_l"]) == eval_l
         and row["state"] == state and int(row["segment"]) == segment + 1),
        key=lambda row: int(row["lag_l_updates"]),
    )
    selected_lags = np.asarray([float(row["lag_l_updates"]) for row in selected])
    if not np.array_equal(selected_lags, lags):
        raise AssertionError(f"Cluster-bootstrap lag grid mismatch for {condition}/L{eval_l}/{state}/segment{segment + 1}.")
    return (np.asarray([float(row["cluster_ci95_low"]) for row in selected]),
            np.asarray([float(row["cluster_ci95_high"]) for row in selected]))


def plot_positive(axis: Any, lags: np.ndarray, mean: np.ndarray, low: np.ndarray, high: np.ndarray, color: Any,
                  label: str, seed_curves: list[np.ndarray] | None = None) -> None:
    positive = mean > 0
    if seed_curves:
        for seed_curve in seed_curves:
            positive_seed = seed_curve > 0
            if np.any(positive_seed):
                axis.plot(lags[positive_seed], seed_curve[positive_seed], color=color, linewidth=.7, alpha=.28)
    if np.any(positive):
        axis.plot(lags[positive], mean[positive], color=color, linewidth=1.6, label=label)
        axis.fill_between(lags[positive], np.maximum(low[positive], np.finfo(float).tiny), high[positive], color=color, alpha=.12)
    if np.any(~positive):
        axis.text(.02, .04, "× exact zero: no H update", transform=axis.transAxes, fontsize=7, color=color)


def segment_title(segment: int, args: Any) -> str:
    if getattr(args, "min_outer_cycles", None):
        return f"outer-progress segment {segment + 1}"
    return f"origins {PHYSICAL_BOUNDARIES[segment]}–{PHYSICAL_BOUNDARIES[segment + 1]}"


def unit_boundaries(metadata: list[dict[str, str]], condition: str, eval_l: int) -> np.ndarray:
    rows = [row for row in metadata if row["condition"] == condition and int(row["eval_l"]) == eval_l]
    if not rows:
        raise KeyError(f"Missing metadata for {condition}/L{eval_l}.")
    boundaries = np.asarray(json.loads(rows[0]["segment_boundaries_l_updates"]), dtype=np.int64)
    if any(not np.array_equal(boundaries, np.asarray(json.loads(row["segment_boundaries_l_updates"]), dtype=np.int64)) for row in rows[1:]):
        raise AssertionError(f"Seed segment boundaries differ for {condition}/L{eval_l}.")
    return boundaries


def create_figures(output_dir: Path, args: Any) -> None:
    metadata = read_csv(output_dir / "rollout_metadata.csv")
    seed_summary = read_csv(output_dir / "msd_seed_cluster_bootstrap.csv")
    evals = DEFAULT_EVAL_L_VALUES
    for condition in H2L6_CONDITIONS:
        figure, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey="row")
        for row, state in enumerate(("h", "l")):
            for segment in range(4):
                axis = axes[row, segment]
                for index, eval_l in enumerate(evals):
                    lags, mean, _low, _high, per_seed = curve(metadata, condition, eval_l, state, segment)
                    low, high = cluster_bounds(seed_summary, condition, eval_l, state, segment, lags)
                    plot_positive(axis, lags, mean, low, high, L_DEPTH_COLORS[eval_l], f"L={eval_l}", per_seed)
                axis.set_xscale("log", base=2); axis.set_yscale("log", base=2); axis.grid(True, which="both", alpha=.2)
                axis.set_title(f"{state.upper()}, {segment_title(segment, args)}")
                if segment == 0: axis.set_ylabel("per-coordinate MSD")
                if row == 1: axis.set_xlabel("lag (underlying L updates)")
        axes[0, -1].legend(fontsize=7, ncol=2)
        clock = "outer-progress" if getattr(args, "min_outer_cycles", None) else "physical-time"
        figure.suptitle(f"{clock.capitalize()} L-depth MSD — {condition}")
        figure.tight_layout()
        for suffix in ("png", "pdf"): figure.savefig(output_dir / f"msd_l_depth_{condition}.{suffix}", dpi=200)
        plt.close(figure)
    figure, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True, sharey="row")
    for row, condition in enumerate(H2L6_CONDITIONS):
        for segment in range(4):
            axis = axes[row, segment]
            for index, eval_l in enumerate(evals):
                lags, mean, _low, _high, per_seed = curve(metadata, condition, eval_l, "hl_concat", segment)
                low, high = cluster_bounds(seed_summary, condition, eval_l, "hl_concat", segment, lags)
                plot_positive(axis, lags, mean, low, high, L_DEPTH_COLORS[eval_l], f"L={eval_l}", per_seed)
            for control, style in (("H2L1_h", "--"), ("RT", ":")):
                state = "hl_concat" if control == "H2L1_h" else "rt"
                lags, mean, _low, _high, control_seeds = curve(metadata, control, 1, state, segment)
                for control_seed in control_seeds:
                    axis.plot(lags, control_seed, style, color="#222222" if control == "H2L1_h" else "#777777", linewidth=.6, alpha=.25)
                axis.plot(lags, mean, style, color="#222222" if control == "H2L1_h" else "#777777", linewidth=1.2, label=control)
            axis.set_xscale("log", base=2); axis.set_yscale("log", base=2); axis.grid(True, which="both", alpha=.2)
            axis.set_title(f"{condition}, {segment_title(segment, args)}")
            if segment == 0: axis.set_ylabel("joint / RT MSD")
            if row == 2: axis.set_xlabel("lag (underlying L updates)")
    axes[0, -1].legend(fontsize=7, ncol=2)
    clock = "outer-progress" if getattr(args, "min_outer_cycles", None) else "physical-time"
    figure.suptitle(f"Joint HRM state versus fixed H2L1-H and RT controls ({clock})")
    figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"joint_l_depth_controls.{suffix}", dpi=200)
    plt.close(figure)
    create_per_model_state_figures(output_dir, metadata, seed_summary, args)
    create_ratio_figure(output_dir, metadata, args)


def create_per_model_state_figures(output_dir: Path, metadata: list[dict[str, str]],
                                   seed_summary: list[dict[str, str]], args: Any) -> None:
    """One 2×2 H/L/H+L/[H,L] figure for every trained-model/evaluation-L pair."""
    cmap = plt.get_cmap("tab10")
    configurations = [(condition, eval_l) for condition in H2L6_CONDITIONS for eval_l in DEFAULT_EVAL_L_VALUES]
    configurations.append(("H2L1_h", 1))
    for condition, eval_l in configurations:
        boundaries = unit_boundaries(metadata, condition, eval_l)
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for axis, state in zip(axes.ravel(), STATE_NAMES):
            for segment in range(4):
                lags, mean, _low, _high, per_seed = curve(metadata, condition, eval_l, state, segment)
                low, high = cluster_bounds(seed_summary, condition, eval_l, state, segment, lags)
                start, end = boundaries[segment], boundaries[segment + 1]
                plot_positive(axis, lags, mean, low, high, cmap(segment), f"{start}–{end}", per_seed)
            axis.set_xscale("log", base=2); axis.set_yscale("log", base=2); axis.grid(True, which="both", alpha=.2)
            axis.set_title({"h": "H", "l": "L", "h_plus_l": "H+L", "hl_concat": "[H,L]"}[state])
            axis.set_xlabel("lag (underlying L updates)")
            axis.set_ylabel("per-coordinate MSD")
        axes[0, 1].legend(title="origin window", fontsize=8)
        clock = "outer-progress windows" if getattr(args, "min_outer_cycles", None) else "physical-time windows"
        figure.suptitle(f"All-state MSD ({clock}) — {condition}, inference L={eval_l}")
        figure.tight_layout()
        stem = f"all_state_msd_{condition}_evalL{eval_l}"
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{stem}.{suffix}", dpi=200)
        plt.close(figure)


def ratio_rows(metadata: list[dict[str, str]]) -> list[dict[str, object]]:
    """Calculate seed-level R=MSD_H/MSD_L while retaining exact H zeros."""
    rows: list[dict[str, object]] = []
    for meta in metadata:
        if meta["kind"] not in ("hrm", "trm"):
            continue
        with np.load(meta["per_puzzle_file"], allow_pickle=False) as arrays:
            names = [str(value) for value in arrays["state_names"]]
            h_index, l_index = names.index("h"), names.index("l")
            values = arrays["msd"].astype(np.float64)
            for segment in range(4):
                for point, lag in enumerate(arrays["lag_l_updates"]):
                    h = values[:, h_index, segment, point]
                    l = values[:, l_index, segment, point]
                    if not (np.isfinite(h).all() and np.isfinite(l).all()):
                        continue
                    h_mean, l_mean = float(h.mean()), float(l.mean())
                    if l_mean == 0.0:
                        ratio, status = math.nan, "zero L MSD"
                    elif h_mean == 0.0:
                        ratio, status = 0.0, "no H update"
                    else:
                        ratio, status = h_mean / l_mean, "defined"
                    rows.append({
                        "condition": meta["condition"], "readout": meta["readout"],
                        "eval_l": int(meta["eval_l"]), "seed": int(meta["seed"]),
                        "segment": segment + 1, "lag_l_updates": int(lag),
                        "msd_h": h_mean, "msd_l": l_mean, "h_over_l": ratio, "status": status,
                    })
    return rows


def ratio_curve(rows: list[dict[str, str]], condition: str, eval_l: int, segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["condition"] == condition and int(row["eval_l"]) == eval_l
         and int(row["segment"]) == segment + 1 and row["status"] != "zero L MSD"),
        key=lambda row: (int(row["seed"]), int(row["lag_l_updates"])),
    )
    by_seed: dict[int, list[dict[str, str]]] = {}
    for row in selected:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if not by_seed:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    per_seed = [sorted(group, key=lambda row: int(row["lag_l_updates"])) for _, group in sorted(by_seed.items())]
    lags = np.asarray([float(row["lag_l_updates"]) for row in per_seed[0]])
    values = np.asarray([[float(row["h_over_l"]) for row in group] for group in per_seed])
    if values.ndim != 2 or any(np.asarray([float(row["lag_l_updates"]) for row in group]).shape != lags.shape for group in per_seed):
        raise AssertionError("Seed ratio lag grids differ.")
    return lags, values.mean(axis=0), np.quantile(values, .025, axis=0), np.quantile(values, .975, axis=0)


def create_ratio_figure(output_dir: Path, metadata: list[dict[str, str]], args: Any) -> None:
    rows = [dict(row) for row in ratio_rows(metadata)]
    atomic_csv(output_dir / "h_over_l_ratio.csv", rows, RATIO_FIELDS)
    figure, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True, sharey=True)
    for row, condition in enumerate(H2L6_CONDITIONS):
        for segment in range(4):
            axis = axes[row, segment]
            zero_labels: list[str] = []
            for index, eval_l in enumerate(DEFAULT_EVAL_L_VALUES):
                lags, mean, low, high = ratio_curve(rows, condition, eval_l, segment)
                if not len(lags):
                    continue
                positive = mean > 0
                color = L_DEPTH_COLORS[eval_l]
                if np.any(positive):
                    axis.plot(lags[positive], mean[positive], color=color, linewidth=1.6, label=f"L={eval_l}")
                    axis.fill_between(lags[positive], np.maximum(low[positive], np.finfo(float).tiny), high[positive], color=color, alpha=.12)
                if np.any(~positive):
                    zero_labels.append(f"L={eval_l}")
            # RT has no H/L decomposition, so only H2L1-H is a ratio control.
            lags, mean, _low, _high = ratio_curve(rows, "H2L1_h", 1, segment)
            positive = mean > 0
            if np.any(positive):
                axis.plot(lags[positive], mean[positive], "--", color="black", linewidth=1.2, label="H2L1-H")
            axis.axhline(1, color="0.35", linewidth=.8, linestyle=":")
            axis.set_xscale("log", base=2); axis.set_yscale("log", base=2); axis.grid(True, which="both", alpha=.2)
            axis.set_title(f"{condition}, {segment_title(segment, args)}")
            if zero_labels:
                axis.text(.02, .04, "no H update: " + ", ".join(zero_labels), transform=axis.transAxes, fontsize=6.5)
            if segment == 0: axis.set_ylabel("R(lag) = MSD_H / MSD_L")
            if row == 2: axis.set_xlabel("lag (underlying L updates)")
    axes[0, -1].legend(fontsize=7, ncol=2)
    clock = "outer-progress segments" if getattr(args, "min_outer_cycles", None) else "physical L-update clock"
    figure.suptitle(f"Full-lag H/L mobility ratio ({clock})")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"h_over_l_l_depth.{suffix}", dpi=200)
    plt.close(figure)


def write_readme(output_dir: Path, args: Any) -> None:
    if getattr(args, "min_outer_cycles", None):
        description = f"""# Core-five L-depth long rollout (minimum {args.min_outer_cycles} outer cycles)

This compute-unmatched companion profile uses
`max(4096, 2 * {args.min_outer_cycles} * inference_L)` lower updates for every
HRM run. Since H=2, this guarantees at least {args.min_outer_cycles} ordinary
outer calls even for L=512/1024 (16,384/32,768 lower updates respectively).
Its four windows are equally spaced in completed ordinary outer calls, then
stored as lower-update endpoints in `rollout_metadata.csv`. They should not be
interpreted as shared absolute physical-time windows across L values.
"""
    else:
        description = """# Core-five physical-time L-depth long rollout

This is separate from `core_five_long_rollout/`: it reuses that analysis's selected
native-L6 checkpoints but does not reselect epochs. H2L6-H/L/HL run inference L
depths 6, 8, 16, 32, 64, 128, 256, 512, and 1024; H2L1-H and RT are fixed controls.
All conditions have exactly 4,096 underlying L/recurrent updates and share the
physical windows `[0,48)`, `[48,192)`, `[192,768)`, `[768,4096)`.

HRM states are sampled after every L update. H is therefore exactly unchanged
between scheduled H updates. H2L1024 has only four H updates; exact zero H MSD in
some physical-time comparisons is intentional and means no H update occurred, not
missing data. H2L6 ends with 682 completed H updates and a four-L-update tail.
"""
    (output_dir / "README.md").write_text(description + """

`h_over_l_ratio.csv` and `h_over_l_l_depth.png/pdf` report
`R(lag)=MSD_H(lag)/MSD_L(lag)`. Exact-zero H MSD is retained as zero in the CSV
with status `no H update`; it is annotated rather than replaced by an artificial
positive value on the log-scale plot. RT has no H/L ratio and is not plotted there.

`all_state_msd_<condition>_evalL<k>.png/pdf` gives a compact 2×2 view of H, L,
H+L, and [H,L] for each HRM model/inference-L pair. Colours are the four recorded
origin windows; faint curves are individual training seeds, solid
curves their mean, and bands are the cluster-bootstrap 95% intervals.
""")


def merge(args: Any) -> None:
    workers = args.merge_from
    hashes = [(Path(worker) / "sample_manifest.sha256").read_text().strip() for worker in workers]
    if len(set(hashes)) != 1:
        raise ValueError("Worker manifests differ; refusing to merge incomparable samples.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, str]] = []
    selections: dict[tuple[str, int], dict[str, str]] = {}
    for worker in workers:
        metadata.extend(read_csv(Path(worker) / "rollout_metadata.csv"))
        for row in read_csv(Path(worker) / "best_checkpoints.csv"):
            selections[(row["condition"], int(row["seed"]))] = row
        for source in (Path(worker) / "per_puzzle_msd").glob("*.npz"):
            target = args.output_dir / "per_puzzle_msd" / source.name
            if target.exists():
                if not filecmp.cmp(source, target, shallow=False):
                    raise ValueError(f"Conflicting per-puzzle result: {target.name}")
                continue  # Resume a merge interrupted after this file was copied.
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    expected = {unit.key for unit in args.all_units}
    actual = {(row["condition"], int(row["seed"]), int(row["eval_l"])) for row in metadata}
    if actual != expected:
        raise ValueError(f"Incomplete merge: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    for row in metadata:
        row["per_puzzle_file"] = str(args.output_dir / "per_puzzle_msd" / Path(row["per_puzzle_file"]).name)
    atomic_csv(args.output_dir / "best_checkpoints.csv", list(selections.values()), SELECTION_FIELDS)
    atomic_csv(args.output_dir / "rollout_metadata.csv", metadata, METADATA_FIELDS)
    shutil.copy2(Path(workers[0]) / "sample_manifest.csv", args.output_dir / "sample_manifest.csv")
    shutil.copy2(Path(workers[0]) / "sample_manifest.sha256", args.output_dir / "sample_manifest.sha256")
    finalize(args)


def finalize(args: Any) -> None:
    puzzle_rows, seed_rows = summarize(args.output_dir, args.bootstrap_replicates, args.sample_seed)
    atomic_csv(args.output_dir / "msd_puzzle_bootstrap.csv", puzzle_rows, SUMMARY_FIELDS)
    atomic_csv(args.output_dir / "msd_seed_cluster_bootstrap.csv", seed_rows, SEED_FIELDS)
    create_figures(args.output_dir, args)
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({
        "profile": "core-five-l-depth-min-outer16" if getattr(args, "min_outer_cycles", None) else "core-five-l-depth",
        "eval_l_values": args.l_depth_values, "samples": args.samples,
        "sample_seed": args.sample_seed, "max_l_updates": int(PHYSICAL_BOUNDARIES[-1]),
        "physical_windows": PHYSICAL_BOUNDARIES.tolist(), "lag_points": args.lag_points,
        "min_outer_cycles": getattr(args, "min_outer_cycles", None),
        "analysis_scheme": MIN_OUTER_SCHEME if getattr(args, "min_outer_cycles", None) else SCHEME,
    }, indent=2) + "\n")
    write_readme(args.output_dir, args)


def main_l_depth(args: Any) -> None:
    discovered = core_runs(args.checkpoints_root, args.core_seeds)
    args.all_units = units_from_runs(discovered, args.core_seeds, args.l_depth_values)
    if args.merge_from:
        merge(args); return
    units = args.all_units[args.shard_index::args.num_shards] if args.shard_index is not None else args.all_units
    if not units: raise ValueError("This shard received no sweep units.")
    reference = args.reference_best_checkpoints
    selected = selected_checkpoints(reference, args.all_units)
    first_config = units[0].run.config
    create_dataloader = load_module(f"dataset.{first_config.data.name}@create_dataloader")
    loader, metadata = create_dataloader("test_hard", first_config.local_batch_size, rank=0, world_size=1, **data_kwargs(first_config))
    fixed_x, indices = fixed_random_samples(loader, args.samples, args.sample_seed)
    manifest = write_manifest(args.output_dir, indices, args.sample_seed)
    chunks = math.ceil(len(fixed_x) / args.rollout_batch_size)
    specs = {unit.key: (*rollout_spec(unit, args),) for unit in units}
    progress = tqdm(total=sum(len(lags_for_boundaries(boundaries, args.lag_points)) * chunks
                              for _updates, boundaries, _scheme in specs.values()),
                    desc="Core L-depth paired rollouts", unit="lag-batch")
    rows = read_csv(args.output_dir / "rollout_metadata.csv")
    completed = {(row["condition"], int(row["seed"]), int(row["eval_l"])) for row in rows}
    for unit in units:
        progress.set_postfix_str(unit.label)
        updates, boundaries, scheme = specs[unit.key]
        lags = lags_for_boundaries(boundaries, args.lag_points)
        if unit.key in completed:
            progress.update(len(lags) * chunks); continue
        row = collect_unit(unit, selected[(unit.run.condition, unit.run.seed)], fixed_x, metadata, args,
                           manifest, lags, updates, boundaries, scheme, progress)
        rows = [old for old in rows if (old["condition"], int(old["seed"]), int(old["eval_l"])) != unit.key] + [row]
        atomic_csv(args.output_dir / "rollout_metadata.csv", rows, METADATA_FIELDS)
        atomic_csv(args.output_dir / "best_checkpoints.csv", list(selected.values()), SELECTION_FIELDS)
    progress.close()
    if args.num_shards == 1:
        finalize(args)
