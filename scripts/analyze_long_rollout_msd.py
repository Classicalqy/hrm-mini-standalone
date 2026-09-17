#!/usr/bin/env python
"""Select sweep checkpoints and measure deterministic full-state long-rollout MSD.

The script is deliberately independent from the accuracy/L-ablation evaluator.  It
re-evaluates every saved epoch on native test_hard inference, selects the best epoch
per condition and seed, then computes long-rollout MSD directly from full latent
states.  No W&B data, trajectory files, random projections, or dimensionality
reduction are used.

Example (run on one GPU):
    python scripts/analyze_long_rollout_msd.py \
        --checkpoints-root checkpoints/h2_l_readout_sweep \
        --rt-checkpoint-dir checkpoints/h2_l_readout_rt_sweep/RT/seed_1

The exact full-state estimator is compute-intensive by design: each requested lag
uses a pair of synchronized deterministic rollouts, so only two carries are held in
memory and no large trajectory is written to disk.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Iterable, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from arch.layers import Carry
from scripts.long_rollout_msd_utils import (
    STATE_NAMES,
    log_spaced_lags,
    rt_state_msd,
    segment_boundaries,
    segment_for_pair,
    segment_lags,
    state_msd,
)

if TYPE_CHECKING:
    from train import TrainConfig


def load_module(path: str) -> Any:
    """Import training machinery only for workflows that actually evaluate models."""
    from train import load_module as train_load_module

    return train_load_module(path)


def run_inference(*args: Any, **kwargs: Any) -> Any:
    """Keep plots-only independent from optional training-time dependencies."""
    from train import run_inference as train_run_inference

    return train_run_inference(*args, **kwargs)


H_CYCLES = 2
STATE_TITLES = {"h": "H", "l": "L", "h_plus_l": "H+L", "hl_concat": "[H,L]"}
HRM_DIRECTORY = re.compile(r"H2L(?P<l_cycles>\d+)_(?P<readout>h|l|hl)$")
SEED_DIRECTORY = re.compile(r"seed_(?P<seed>\d+)$")
EPOCH_FILE = re.compile(r"epoch_(?P<epoch>\d+)\.pt$")
BEST_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "epoch", "checkpoint",
    "test_exact_match", "cell_accuracy", "selection_started",
]
MSD_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "best_epoch", "checkpoint", "state", "segment",
    "segment_start_block", "segment_end_block", "segment_start_l_update", "segment_end_l_update",
    "lag_blocks", "lag_l_updates", "msd", "origins",
]
SUMMARY_FIELDS = [
    "kind", "condition", "readout", "train_l", "state", "segment", "segment_start_block", "segment_end_block",
    "segment_start_l_update", "segment_end_l_update", "lag_blocks", "lag_l_updates", "seeds", "msd_mean", "msd_ci95", "origins_per_seed",
]
ROLLOUT_FIELDS = [
    "kind", "condition", "readout", "train_l", "seed", "best_epoch", "checkpoint", "samples",
    "requested_l_updates", "actual_l_updates", "total_blocks", "segment_boundaries_blocks", "lag_blocks", "state_shape",
]


@dataclass(frozen=True)
class RunDirectory:
    """A single seed's collection of epoch checkpoints."""

    kind: Literal["hrm", "trm", "rt"]
    condition: str
    seed: int
    directory: Path
    config: TrainConfig
    l_cycles: int | None
    readout: str


def data_kwargs(config: TrainConfig) -> dict[str, Any]:
    kwargs = dict(config.data.__pydantic_extra__ or {})
    kwargs.update(augment=False, repeat=1)
    return kwargs


def load_config(directory: Path) -> TrainConfig:
    from train import TrainConfig

    path = directory / "model_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model config: {path}")
    with path.open() as handle:
        return TrainConfig(**yaml.safe_load(handle))


def epoch_checkpoints(directory: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in directory.glob("epoch_*.pt"):
        match = EPOCH_FILE.fullmatch(path.name)
        if match:
            checkpoints.append((int(match["epoch"]), path))
    return sorted(checkpoints)


def discover_hrm_runs(root: Path) -> list[RunDirectory]:
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    runs: list[RunDirectory] = []
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        condition_match = HRM_DIRECTORY.fullmatch(condition_dir.name)
        if condition_match is None:
            continue
        named_l = int(condition_match["l_cycles"])
        named_readout = condition_match["readout"]
        for seed_dir in sorted(path for path in condition_dir.iterdir() if path.is_dir()):
            seed_match = SEED_DIRECTORY.fullmatch(seed_dir.name)
            if seed_match is None or not epoch_checkpoints(seed_dir):
                continue
            config = load_config(seed_dir)
            arch = config.arch.__pydantic_extra__ or {}
            if config.arch.name != "hrm@HRM":
                raise ValueError(f"{seed_dir} is not a standard HRM checkpoint ({config.arch.name}).")
            if arch.get("H_cycles") != H_CYCLES:
                raise ValueError(f"{seed_dir} has H_cycles={arch.get('H_cycles')}, expected {H_CYCLES}.")
            if arch.get("L_cycles") != named_l or arch.get("readout") != named_readout:
                raise ValueError(f"Directory/config mismatch in {seed_dir}.")
            runs.append(RunDirectory(
                kind="hrm", condition=condition_dir.name, seed=int(seed_match["seed"]),
                directory=seed_dir, config=config, l_cycles=named_l, readout=named_readout,
            ))
    if not runs:
        raise FileNotFoundError(f"No H2L{{L}}_{{h,l,hl}}/seed_*/epoch_*.pt runs found under {root}.")
    return runs


def discover_rt_run(directory: Path) -> RunDirectory:
    config = load_config(directory)
    if config.arch.name != "rt@RecurrentTransformer":
        raise ValueError(f"{directory} is not an RT checkpoint directory ({config.arch.name}).")
    if not epoch_checkpoints(directory):
        raise FileNotFoundError(f"No epoch_*.pt files in {directory}.")
    seed_match = SEED_DIRECTORY.fullmatch(directory.name)
    if seed_match is None:
        raise ValueError("--rt-checkpoint-dir must be a seed_N directory.")
    return RunDirectory("rt", "RT", int(seed_match["seed"]), directory, config, None, "rt")


def discover_rt_runs(root: Path) -> list[RunDirectory]:
    """Discover the standard RT baseline stored alongside the H2L sweep."""
    rt_root = root / "RT"
    if not rt_root.is_dir():
        return []
    runs = []
    for seed_dir in sorted(path for path in rt_root.iterdir() if path.is_dir()):
        if SEED_DIRECTORY.fullmatch(seed_dir.name) is not None and epoch_checkpoints(seed_dir):
            runs.append(discover_rt_run(seed_dir))
    return runs


def build_model(run: RunDirectory, checkpoint: Path, metadata: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cls = load_module(f"arch.{run.config.arch.name}")
    model = model_cls((run.config.arch.__pydantic_extra__ or {}) | metadata).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def native_accuracy(
    model: torch.nn.Module, run: RunDirectory, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device,
) -> tuple[float, float]:
    """Return training-consistent exact match and BOS-excluded cell accuracy."""
    if run.kind == "hrm":
        model.H_cycles = H_CYCLES  # type: ignore[attr-defined]
        model.L_cycles = run.l_cycles  # type: ignore[attr-defined]
    total_exact = total_examples = total_cells = total_correct_cells = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        carry: Carry = model.initial_carry  # type: ignore[attr-defined]
        predictions = None
        for _ in range(run.config.cycles_per_data):
            carry, predictions = run_inference(model, carry, x)
        assert predictions is not None
        total_exact += torch.all(predictions == y, dim=-1).sum().item()
        total_examples += y.shape[0]
        total_correct_cells += (predictions[:, 1:] == y[:, 1:]).sum().item()
        total_cells += y[:, 1:].numel()
    if total_examples == 0:
        raise RuntimeError("test_hard loader produced no examples.")
    return total_exact / total_examples, total_correct_cells / total_cells


def select_best_checkpoint(
    run: RunDirectory,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    metadata: dict[str, Any],
    device: torch.device,
    progress: tqdm[Any],
) -> dict[str, object]:
    candidates = epoch_checkpoints(run.directory)
    best: dict[str, object] | None = None
    for epoch, checkpoint in candidates:
        progress.set_postfix_str(f"{run.condition}/seed_{run.seed}, epoch={epoch}")
        model = build_model(run, checkpoint, metadata, device)
        exact_match, cell_accuracy = native_accuracy(model, run, loader, device)
        candidate = {
            "kind": run.kind, "condition": run.condition, "readout": run.readout,
            "train_l": "" if run.l_cycles is None else run.l_cycles,
            "seed": run.seed, "epoch": epoch, "checkpoint": str(checkpoint),
            "test_exact_match": exact_match, "cell_accuracy": cell_accuracy,
        }
        # Epochs are sorted, so strictly greater keeps the earlier checkpoint on ties.
        if best is None or exact_match > float(best["test_exact_match"]):
            best = candidate
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress.update(1)
    assert best is not None
    return best


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_best_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing worker output: {path}")
    with path.open(newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def initial_hrm_state(model: torch.nn.Module, carry: Carry, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_h, z_l = carry["z_H"], carry["z_L"]
    target_shape = (*x.shape, z_h.shape[-1])
    if z_h.ndim < 3:
        z_h = z_h.reshape(1, 1, -1).expand(target_shape)
    if z_l.ndim < 3:
        z_l = z_l.reshape(1, 1, -1).expand(target_shape)
    return z_h, z_l


@torch.inference_mode()
def advance_hrm_block(
    model: torch.nn.Module, carry: Carry, x: torch.Tensor,
) -> tuple[Carry, tuple[torch.Tensor, torch.Tensor]]:
    state: tuple[torch.Tensor, torch.Tensor] | None = None

    def callback(event: str, z_h: torch.Tensor, z_l: torch.Tensor) -> None:
        nonlocal state
        if event != "h":
            raise AssertionError(f"Expected only H trace events, received {event!r}.")
        state = (z_h, z_l)

    carry, _ = model.forward_with_trace(carry, x, callback, events=("h",))  # type: ignore[attr-defined]
    if state is None:
        raise RuntimeError("HRM H-block trace emitted no state.")
    return carry, state


@torch.inference_mode()
def advance_rt_step(model: torch.nn.Module, carry: Carry, x: torch.Tensor) -> tuple[Carry, torch.Tensor]:
    state: torch.Tensor | None = None

    def callback(event: str, z: torch.Tensor) -> None:
        nonlocal state
        if event != "z":
            raise AssertionError(f"Expected only RT trace events, received {event!r}.")
        state = z

    carry, _ = model.forward_with_trace(carry, x, callback, events=("z",))  # type: ignore[attr-defined]
    if state is None:
        raise RuntimeError("RT trace emitted no state.")
    return carry, state


def trajectory_rows_hrm(
    model: torch.nn.Module,
    run: RunDirectory,
    x: torch.Tensor,
    max_l_updates: int,
    lag_points: int,
    progress: tqdm[Any],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    assert run.l_cycles is not None
    native_l = run.l_cycles
    total_blocks = max_l_updates // native_l
    boundaries = segment_boundaries(total_blocks)
    lags = segment_lags(boundaries, lag_points)
    if not lags:
        raise RuntimeError("No valid lags for requested rollout.")

    original_h_cycles = model.H_cycles  # type: ignore[attr-defined]
    model.H_cycles = 1  # type: ignore[attr-defined]
    model.L_cycles = native_l  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []
    try:
        for lag in lags:
            reference_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            leading_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            reference_state = initial_hrm_state(model, reference_carry, x)
            leading_state = initial_hrm_state(model, leading_carry, x)
            for _ in range(lag):
                leading_carry, leading_state = advance_hrm_block(model, leading_carry, x)

            totals = {segment: {state: [0.0, 0] for state in STATE_NAMES} for segment in range(4)}
            for t in range(total_blocks - lag + 1):
                segment = segment_for_pair(t, lag, boundaries)
                if segment is not None:
                    msd = state_msd(*reference_state, *leading_state)
                    for state, value in msd.items():
                        totals[segment][state][0] += value
                        totals[segment][state][1] += 1
                if t < total_blocks - lag:
                    reference_carry, reference_state = advance_hrm_block(model, reference_carry, x)
                    leading_carry, leading_state = advance_hrm_block(model, leading_carry, x)
            for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                for state, (total, count) in totals[segment].items():
                    if count:
                        rows.append({
                            "kind": "hrm", "condition": run.condition, "readout": run.readout,
                            "train_l": native_l, "seed": run.seed, "state": state,
                            "segment": segment + 1, "segment_start_block": start, "segment_end_block": end,
                            "segment_start_l_update": start * native_l, "segment_end_l_update": end * native_l,
                            "lag_blocks": lag, "lag_l_updates": lag * native_l,
                            "msd": total / count, "origins": count,
                        })
            progress.update(1)
    finally:
        model.H_cycles = original_h_cycles  # type: ignore[attr-defined]
    metadata = {
        "kind": "hrm", "condition": run.condition, "readout": run.readout, "train_l": native_l,
        "seed": run.seed, "requested_l_updates": max_l_updates,
        "actual_l_updates": total_blocks * native_l, "total_blocks": total_blocks,
        "segment_boundaries_blocks": json.dumps(boundaries), "lag_blocks": json.dumps(lags),
        "state_shape": "x".join(str(part) for part in initial_hrm_state(model, model.initial_carry, x)[0].shape),
    }
    return rows, metadata


def trajectory_rows_rt(
    model: torch.nn.Module,
    run: RunDirectory,
    x: torch.Tensor,
    max_updates: int,
    lag_points: int,
    progress: tqdm[Any],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    boundaries = segment_boundaries(max_updates)
    lags = segment_lags(boundaries, lag_points)
    original_cycles = model.cycles  # type: ignore[attr-defined]
    model.cycles = 1  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []
    try:
        for lag in lags:
            reference_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            leading_carry: Carry = model.initial_carry  # type: ignore[attr-defined]
            initial = reference_carry["z"].reshape(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
            reference_state = initial
            leading_state = initial
            for _ in range(lag):
                leading_carry, leading_state = advance_rt_step(model, leading_carry, x)
            totals = {segment: [0.0, 0] for segment in range(4)}
            for t in range(max_updates - lag + 1):
                segment = segment_for_pair(t, lag, boundaries)
                if segment is not None:
                    totals[segment][0] += rt_state_msd(reference_state, leading_state)
                    totals[segment][1] += 1
                if t < max_updates - lag:
                    reference_carry, reference_state = advance_rt_step(model, reference_carry, x)
                    leading_carry, leading_state = advance_rt_step(model, leading_carry, x)
            for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                total, count = totals[segment]
                if count:
                    rows.append({
                        "kind": "rt", "condition": "RT", "readout": "rt", "train_l": "", "seed": run.seed,
                        "state": "rt", "segment": segment + 1, "segment_start_block": start,
                        "segment_end_block": end, "segment_start_l_update": start,
                        "segment_end_l_update": end, "lag_blocks": lag, "lag_l_updates": lag,
                        "msd": total / count, "origins": count,
                    })
            progress.update(1)
    finally:
        model.cycles = original_cycles  # type: ignore[attr-defined]
    metadata = {
        "kind": "rt", "condition": "RT", "readout": "rt", "train_l": "", "seed": run.seed,
        "requested_l_updates": max_updates, "actual_l_updates": max_updates, "total_blocks": max_updates,
        "segment_boundaries_blocks": json.dumps(boundaries), "lag_blocks": json.dumps(lags),
        "state_shape": f"{x.shape[0]}x{x.shape[1]}x{model.initial_carry['z'].shape[-1]}",
    }
    return rows, metadata


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = {}
    origins: dict[tuple[object, ...], int] = {}
    fields = ("kind", "condition", "readout", "train_l", "state", "segment", "segment_start_block", "segment_end_block", "segment_start_l_update", "segment_end_l_update", "lag_blocks", "lag_l_updates")
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(float(row["msd"]))
        origins[key] = int(row["origins"])
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        mean = float(np.mean(values))
        if len(values) > 1:
            ci95 = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
        else:
            ci95 = 0.0
        summary = dict(zip(fields, key))
        summary.update(seeds=len(values), msd_mean=mean, msd_ci95=ci95, origins_per_seed=origins[key])
        summaries.append(summary)
    return summaries


def plot_condition(output_dir: Path, condition: str, rows: list[dict[str, object]]) -> None:
    is_rt = rows[0]["kind"] == "rt"
    states = ("rt",) if is_rt else STATE_NAMES
    figure, axes = plt.subplots(1, 1, figsize=(7, 5)) if is_rt else plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes_list = [axes] if is_rt else list(axes.flat)
    for axis, state in zip(axes_list, states):
        state_rows = [row for row in rows if row["state"] == state]
        for segment in range(1, 5):
            points = sorted((row for row in state_rows if int(row["segment"]) == segment), key=lambda row: int(row["lag_l_updates"]))
            if not points:
                continue
            x = np.array([float(row["lag_l_updates"]) for row in points])
            y = np.array([float(row["msd_mean"]) for row in points])
            ci = np.array([float(row["msd_ci95"]) for row in points])
            first = points[0]
            label = f"segment {segment}: {first['segment_start_l_update']}–{first['segment_end_l_update']}"
            axis.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=label)
            if np.any(ci > 0):
                axis.fill_between(x, np.maximum(y - ci, np.finfo(float).tiny), y + ci, alpha=0.2)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_title("RT state" if state == "rt" else STATE_TITLES[state])
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes_list:
        axis.set_xlabel("Lag (underlying recurrent/L updates, log₂)")
        axis.set_ylabel("Full-state per-coordinate MSD (log₂)")
    figure.suptitle(f"Deterministic long-rollout MSD — {condition}")
    figure.tight_layout()
    stem = "msd_RT" if is_rt else f"msd_{condition}"
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=200)
    plt.close(figure)


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text("""# Full-state deterministic long-rollout MSD

`best_checkpoints.csv` is produced by re-evaluating every saved epoch at its native
inference schedule on full `test_hard`; no W&B history is used. Ties select the
earlier epoch.

For HRM, states are sampled immediately after each H update. A native H block has
`train_L` lower-level updates, so both the plotted lag and segment boundaries are
reported in underlying L-update units. The four segments are log-uniform in the H
boundary clock: `0, round(N**1/4), round(N**1/2), round(N**3/4), N`, where
`N=floor(max_l_updates/train_L)`. Both axes in the figures use base-2 log scale.

For a complete state tensor `X` (all batch examples, all 82 token positions, and all
hidden dimensions), the estimator is `mean((X[t+lag] - X[t])**2)`, averaged over
valid time origins. It uses full state tensors directly and does not project or save
trajectories. A paired deterministic rollout recomputes each lag while keeping only
two carries in memory.

`[H,L]` denotes direct concatenation. Because the CSV uses *per-coordinate* MSD,
`MSD_[H,L] = (MSD_H + MSD_L) / 2` up to floating-point roundoff. `H+L` instead
includes the H/L displacement alignment cross term. RT is a separate single-state
control and has no synthetic H/L decomposition.
""")


def merge_worker_outputs(input_dirs: list[Path], output_dir: Path) -> None:
    """Merge non-overlapping GPU worker outputs without loading any checkpoints."""
    best_rows: list[dict[str, object]] = []
    msd_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        best_rows.extend(read_rows(input_dir / "best_checkpoints.csv"))
        msd_rows.extend(read_rows(input_dir / "msd_full_state.csv"))
        rollout_rows.extend(read_rows(input_dir / "rollout_metadata.csv"))
    best_keys = [(row["kind"], row["condition"], row["seed"]) for row in best_rows]
    msd_keys = [
        (row["kind"], row["condition"], row["seed"], row["state"], row["segment"], row["lag_blocks"])
        for row in msd_rows
    ]
    if len(best_keys) != len(set(best_keys)) or len(msd_keys) != len(set(msd_keys)):
        raise ValueError("Worker outputs overlap; pass exactly one output directory for each shard.")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_rows.sort(key=lambda row: (str(row["kind"]), str(row["condition"]), int(row["seed"])))
    msd_rows.sort(key=lambda row: (str(row["kind"]), str(row["condition"]), int(row["seed"]), str(row["state"]), int(row["segment"]), int(row["lag_blocks"])))
    rollout_rows.sort(key=lambda row: (str(row["kind"]), str(row["condition"]), int(row["seed"])))
    write_csv(output_dir / "best_checkpoints.csv", best_rows, BEST_FIELDS)
    write_csv(output_dir / "msd_full_state.csv", msd_rows, MSD_FIELDS)
    write_csv(output_dir / "rollout_metadata.csv", rollout_rows, ROLLOUT_FIELDS)
    summary_rows = summarize_rows(msd_rows)
    write_csv(output_dir / "msd_seed_summary.csv", summary_rows, SUMMARY_FIELDS)
    for condition in sorted({str(row["condition"]) for row in summary_rows}):
        plot_condition(output_dir, condition, [row for row in summary_rows if row["condition"] == condition])
    (output_dir / "analysis_metadata.json").write_text(json.dumps({
        "merged_from": [str(path) for path in input_dirs], "full_state": True, "random_projection": False,
    }, indent=2) + "\n")
    write_readme(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("all", "core-five", "core-five-l-depth", "core-five-l-depth-min-outer16", "core-h-l-clock", "k55-core", "k55-l-depth"), default="all", help="all preserves the original sweep; core-five runs the statistical five-condition analysis; core-five-l-depth is fixed-compute; core-five-l-depth-min-outer16 guarantees 16 ordinary H=2 outer calls; core-h-l-clock analyses 160 H updates; k55-core selects epochs using --k55-epoch-policy and k55-l-depth sweeps inference L for the easy/hard K55 models.")
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/h2_l_readout_sweep"))
    parser.add_argument("--output-dir", type=Path, help="Output directory (profile default is used when omitted).")
    parser.add_argument("--rt-checkpoint-dir", type=Path, help="Optional additional RT seed_N directory; RT/seed_* under --checkpoints-root is discovered automatically.")
    parser.add_argument("--samples", type=int, help="Fixed test_hard puzzles used for each rollout (profile default when omitted).")
    parser.add_argument("--max-l-updates", type=int, default=4096, help="Requested underlying L/recurrent updates.")
    parser.add_argument("--lag-points", type=int, default=16, help="Log₂-spaced lag points per time segment.")
    parser.add_argument("--device", default="cuda", help="Torch device for evaluation and rollout.")
    parser.add_argument("--reuse-best-checkpoints", action="store_true", help="Reuse output-dir/best_checkpoints.csv instead of re-evaluating epochs.")
    parser.add_argument("--shard-index", type=int, help="Zero-based condition/seed shard index for one GPU worker.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total independent GPU worker shards.")
    parser.add_argument("--merge-from", type=Path, nargs="+", metavar="WORKER_DIR", help="Merge worker outputs and draw final figures without model evaluation.")
    parser.add_argument("--core-seeds", default="1,2,3", help="Required seed list for --profile core-five.")
    parser.add_argument("--sample-seed", type=int, default=20260823, help="Fixed random test_hard sample seed for --profile core-five.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000, help="Puzzle/cluster bootstrap repetitions for --profile core-five.")
    parser.add_argument("--rollout-batch-size", type=int, default=64, help="Puzzle batch size during core-five rollout collection.")
    parser.add_argument("--l-depth-values", default="6,8,16,32,64,128,256,512,1024", help="Comma-separated H2L6 inference L values for --profile core-five-l-depth.")
    parser.add_argument("--reference-best-checkpoints", "--reference-checkpoints", dest="reference_best_checkpoints", type=Path, default=Path("results/core_five_long_rollout/final_absolute/best_checkpoints.csv"), help="Checkpoint-selection CSV used by best-epoch L-depth profiles (the legacy option name remains supported).")
    parser.add_argument("--k55-split", choices=("matched", "easy", "hard"), default="matched", help="K55 evaluation band. matched uses easy puzzles for easy-trained models and hard puzzles for hard-trained models.")
    parser.add_argument("--k55-epoch-policy", choices=("best", "last"), default="best", help="K55 checkpoint policy. best evaluates every epoch; last uses the highest numbered epoch file without metric-based selection.")
    parser.add_argument("--plots-only", action="store_true", help="For k55-l-depth, regenerate figures from an existing merged output without checkpoints or rollout.")
    args = parser.parse_args()
    if args.plots_only and args.profile != "k55-l-depth":
        parser.error("--plots-only is only supported with --profile k55-l-depth.")
    if args.profile in ("core-five", "core-five-l-depth", "core-five-l-depth-min-outer16", "core-h-l-clock", "k55-core", "k55-l-depth"):
        from scripts.core_five_long_rollout import main_core, parse_seeds
        output_defaults = {
            "core-five": "results/core_five_long_rollout",
            "core-five-l-depth": "results/core_five_l_depth_long_rollout",
            "core-five-l-depth-min-outer16": "results/core_five_l_depth_min_outer16_long_rollout",
            "core-h-l-clock": "results/core_h_l_clock_dynamics",
            "k55-core": "results/k55_last_checkpoints" if args.k55_epoch_policy == "last" else "results/k55_best_checkpoints",
            "k55-l-depth": "results/k55_l_depth_last" if args.k55_epoch_policy == "last" else "results/k55_l_depth",
        }
        args.output_dir = args.output_dir or Path(output_defaults[args.profile])
        args.samples = 256 if args.samples is None else args.samples
        try:
            args.core_seeds = parse_seeds(args.core_seeds)
        except ValueError as error:
            parser.error(str(error))
        if args.samples < 1 or args.max_l_updates < 4 or args.lag_points < 1 or args.rollout_batch_size < 1:
            parser.error("--samples, --max-l-updates, --lag-points, and --rollout-batch-size must be positive (max updates >= 4).")
        if args.bootstrap_replicates < 10:
            parser.error("--bootstrap-replicates must be at least 10.")
        if args.num_shards < 1:
            parser.error("--num-shards must be positive.")
        if args.shard_index is None and args.num_shards != 1:
            parser.error("--num-shards requires --shard-index.")
        if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
            parser.error(f"--shard-index must be in [0, {args.num_shards - 1}].")
        args.device = torch.device(args.device)
        if not args.merge_from and not args.plots_only and args.device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; pass --device cpu only for a very small smoke test.")
        if args.profile == "core-h-l-clock":
            if not args.reference_best_checkpoints.is_absolute():
                args.reference_best_checkpoints = PROJECT_ROOT / args.reference_best_checkpoints
            from scripts.core_h_l_clock_dynamics import main_h_l_clock
            main_h_l_clock(args)
            return
        if args.profile == "k55-core":
            from scripts.k55_msd_profiles import main_k55_core
            main_k55_core(args)
            return
        if args.profile in ("core-five-l-depth", "core-five-l-depth-min-outer16", "k55-l-depth"):
            if args.max_l_updates != 4096:
                parser.error("The L-depth profiles use 4096 as the fixed-compute floor; do not override --max-l-updates.")
            try:
                args.l_depth_values = tuple(sorted({int(value) for value in args.l_depth_values.split(",") if value.strip()}))
            except ValueError as error:
                parser.error(f"--l-depth-values must be comma-separated positive integers: {error}")
            if args.l_depth_values != (6, 8, 16, 32, 64, 128, 256, 512, 1024):
                parser.error("--l-depth-values must be exactly 6,8,16,32,64,128,256,512,1024 for the fixed core-five L-depth protocol.")
            if not args.reference_best_checkpoints.is_absolute():
                args.reference_best_checkpoints = PROJECT_ROOT / args.reference_best_checkpoints
            args.min_outer_cycles = 16 if args.profile == "core-five-l-depth-min-outer16" else None
            if args.profile == "k55-l-depth":
                from scripts.k55_msd_profiles import main_k55_l_depth
                main_k55_l_depth(args)
                return
            from scripts.core_five_l_depth_long_rollout import main_l_depth
            main_l_depth(args)
            return
        main_core(args)
        return
    args.output_dir = args.output_dir or Path("results/long_rollout_dynamics")
    args.samples = 64 if args.samples is None else args.samples
    if args.samples < 1 or args.max_l_updates < 4 or args.lag_points < 1:
        parser.error("--samples, --max-l-updates, and --lag-points must be positive (max updates >= 4).")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive.")
    if args.merge_from:
        if args.shard_index is not None or args.num_shards != 1:
            parser.error("--merge-from cannot be combined with shard options.")
        merge_worker_outputs(args.merge_from, args.output_dir)
        return
    if args.shard_index is None and args.num_shards != 1:
        parser.error("--num-shards requires --shard-index.")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards - 1}].")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; pass --device cpu only for a very small smoke test.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_hrm_runs(args.checkpoints_root)
    runs.extend(discover_rt_runs(args.checkpoints_root))
    if args.rt_checkpoint_dir is not None:
        additional_rt = discover_rt_run(args.rt_checkpoint_dir)
        if (additional_rt.kind, additional_rt.condition, additional_rt.seed) in {
            (run.kind, run.condition, run.seed) for run in runs
        }:
            parser.error(f"RT run {args.rt_checkpoint_dir} duplicates one already found under --checkpoints-root.")
        runs.append(additional_rt)
    if args.shard_index is not None:
        runs = runs[args.shard_index::args.num_shards]
    if not runs:
        parser.error("This shard received no checkpoint runs.")
    # Every run uses the same Sudoku data config; instantiate its deterministic test loader once.
    first_config = runs[0].config
    create_dataloader = load_module(f"dataset.{first_config.data.name}@create_dataloader")
    test_loader, metadata = create_dataloader("test_hard", first_config.local_batch_size, rank=0, world_size=1, **data_kwargs(first_config))

    best_path = args.output_dir / "best_checkpoints.csv"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.reuse_best_checkpoints:
        if not best_path.is_file():
            raise FileNotFoundError(f"--reuse-best-checkpoints requested, but {best_path} does not exist.")
        best_rows = read_best_rows(best_path)
    else:
        total_epochs = sum(len(epoch_checkpoints(run.directory)) for run in runs)
        selection_progress = tqdm(total=total_epochs, desc="Select best checkpoints", unit="epoch")
        best_rows = [select_best_checkpoint(run, test_loader, metadata, device, selection_progress) for run in runs]
        selection_progress.close()
        for row in best_rows:
            row["selection_started"] = started
        write_csv(best_path, best_rows, BEST_FIELDS)

    run_by_key = {(run.kind, run.condition, run.seed): run for run in runs}
    chosen: list[tuple[RunDirectory, Path, dict[str, object]]] = []
    for row in best_rows:
        key = (str(row["kind"]), str(row["condition"]), int(row["seed"]))
        run = run_by_key.get(key)
        if run is None:
            raise ValueError(f"Best-checkpoint row does not match a discovered run: {key}.")
        checkpoint = Path(str(row["checkpoint"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
        chosen.append((run, checkpoint, row))

    # Obtain fixed examples once; test_hard uses a non-shuffled distributed sampler.
    fixed_batches = []
    remaining = args.samples
    for x, _y in test_loader:
        fixed_batches.append(x[:remaining])
        remaining -= fixed_batches[-1].shape[0]
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"Requested {args.samples} samples, but test_hard provided only {args.samples - remaining}.")
    fixed_x = torch.cat(fixed_batches, dim=0).to(device, non_blocking=True)

    estimated_lags = sum(len(segment_lags(segment_boundaries(args.max_l_updates // (run.l_cycles or 1)), args.lag_points)) for run, _, _ in chosen)
    trajectory_progress = tqdm(total=estimated_lags, desc="Full-state paired rollouts", unit="lag")
    msd_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    for run, checkpoint, best_row in chosen:
        trajectory_progress.set_postfix_str(f"{run.condition}/seed_{run.seed}")
        model = build_model(run, checkpoint, metadata, device)
        if run.kind == "hrm":
            rows, rollout = trajectory_rows_hrm(model, run, fixed_x, args.max_l_updates, args.lag_points, trajectory_progress)
        else:
            rows, rollout = trajectory_rows_rt(model, run, fixed_x, args.max_l_updates, args.lag_points, trajectory_progress)
        for row in rows:
            row["best_epoch"] = best_row["epoch"]
            row["checkpoint"] = str(checkpoint)
        rollout["best_epoch"] = best_row["epoch"]
        rollout["checkpoint"] = str(checkpoint)
        rollout["samples"] = args.samples
        msd_rows.extend(rows)
        rollout_rows.append(rollout)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    trajectory_progress.close()

    # Full-state concatenation is an exact algebraic control; fail loudly if a
    # future refactor accidentally changes its normalization.
    by_identity: dict[tuple[object, ...], dict[str, float]] = {}
    for row in msd_rows:
        key = tuple(row[field] for field in ("condition", "seed", "segment", "lag_blocks"))
        by_identity.setdefault(key, {})[str(row["state"])] = float(row["msd"])
    max_identity_error = max(
        abs(values["hl_concat"] - (values["h"] + values["l"]) / 2)
        for values in by_identity.values() if {"h", "l", "hl_concat"} <= values.keys()
    )
    if max_identity_error > 1e-5:
        raise RuntimeError(f"Full-state [H,L] MSD identity failed (max error {max_identity_error:.3e}).")

    summary_rows = summarize_rows(msd_rows)
    write_csv(args.output_dir / "msd_full_state.csv", msd_rows, MSD_FIELDS)
    write_csv(args.output_dir / "msd_seed_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.output_dir / "rollout_metadata.csv", rollout_rows, ROLLOUT_FIELDS)
    for condition in sorted({str(row["condition"]) for row in summary_rows}):
        plot_condition(args.output_dir, condition, [row for row in summary_rows if row["condition"] == condition])
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({
        "samples": args.samples, "max_l_updates": args.max_l_updates, "lag_points_per_segment": args.lag_points,
        "checkpoint_root": str(args.checkpoints_root), "rt_checkpoint_dir": str(args.rt_checkpoint_dir) if args.rt_checkpoint_dir else None,
        "full_state": True, "random_projection": False, "max_hl_concat_identity_error": max_identity_error,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
    }, indent=2) + "\n")
    write_readme(args.output_dir)


if __name__ == "__main__":
    main()
