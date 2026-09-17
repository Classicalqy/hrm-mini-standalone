import sys
import types
import unittest
from types import SimpleNamespace

import torch

coolname = types.ModuleType("coolname")
coolname.generate_slug = lambda _count: "stub"
sys.modules.setdefault("coolname", coolname)

from arch.hrm import HRM
from scripts.analyze_long_rollout_msd import RunDirectory
from scripts.core_five_l_depth_long_rollout import (
    DEFAULT_EVAL_L_VALUES,
    PHYSICAL_BOUNDARIES,
    advance_hrm_l,
    cluster_bounds,
    expected_h_updates,
    initial_hrm_state,
    ratio_curve,
    rollout_spec,
    units_from_runs,
)
from test_hrm_readout import tiny_config


def fake_run(condition: str, seed: int) -> RunDirectory:
    return RunDirectory(
        kind="rt" if condition == "RT" else "hrm", condition=condition, seed=seed,
        directory=None, config=None, l_cycles=None if condition == "RT" else (1 if condition == "H2L1_h" else 6),
        readout="rt" if condition == "RT" else "h",
    )  # type: ignore[arg-type]


class CoreFiveLDepthProfileTest(unittest.TestCase):
    def test_expected_h_updates_and_tail(self) -> None:
        self.assertEqual(expected_h_updates(6), (682, 4))
        self.assertEqual(expected_h_updates(8), (512, 0))
        self.assertEqual(expected_h_updates(1024), (4, 0))
        self.assertEqual(PHYSICAL_BOUNDARIES.tolist(), [0, 48, 192, 768, 4096])

    def test_units_cover_all_h2l6_schedules_and_fixed_controls(self) -> None:
        conditions = ("H2L1_h", "H2L6_h", "H2L6_l", "H2L6_hl", "RT")
        runs = [fake_run(condition, seed) for condition in conditions for seed in (1, 2, 3)]
        units = units_from_runs(runs, (1, 2, 3), DEFAULT_EVAL_L_VALUES)
        self.assertEqual(len(units), 87)
        self.assertEqual(sum(unit.run.condition.startswith("H2L6") for unit in units), 81)
        self.assertEqual({unit.eval_l for unit in units if unit.run.condition == "H2L1_h"}, {1})
        self.assertEqual({unit.eval_l for unit in units if unit.run.condition == "RT"}, {1})

    def test_min_outer_profile_guarantees_sixteen_ordinary_h2_outer_calls(self) -> None:
        run = fake_run("H2L6_h", 1)
        args = SimpleNamespace(min_outer_cycles=16)
        updates, boundaries, scheme = rollout_spec(type("Unit", (), {"run": run, "eval_l": 1024})(), args)
        self.assertEqual(updates, 32768)
        self.assertEqual(boundaries.tolist(), [0, 8192, 16384, 24576, 32768])
        self.assertIn("min_outer16", scheme)
        updates, boundaries, _ = rollout_spec(type("Unit", (), {"run": run, "eval_l": 6})(), args)
        self.assertEqual(updates, 4096)
        self.assertEqual(boundaries.tolist(), [0, 1020, 2040, 3060, 4096])

    def test_one_indexed_csv_segments_match_zero_indexed_plot_segments(self) -> None:
        cluster_rows = [
            {"condition": "H2L6_h", "eval_l": "6", "state": "h", "segment": "1", "lag_l_updates": "1",
             "cluster_ci95_low": "0.1", "cluster_ci95_high": "0.2"},
        ]
        low, high = cluster_bounds(cluster_rows, "H2L6_h", 6, "h", 0, torch.tensor([1.]).numpy())
        self.assertEqual(low.tolist(), [.1])
        self.assertEqual(high.tolist(), [.2])
        ratio_rows = [
            {"condition": "H2L6_h", "eval_l": "6", "seed": "1", "segment": "1", "lag_l_updates": "1", "h_over_l": "2", "status": "defined"},
            {"condition": "H2L6_h", "eval_l": "6", "seed": "2", "segment": "1", "lag_l_updates": "1", "h_over_l": "4", "status": "defined"},
        ]
        _lags, mean, _low, _high = ratio_curve(ratio_rows, "H2L6_h", 6, 0)
        self.assertEqual(mean.tolist(), [3.])

    def test_six_step_manual_block_matches_native_hrm_forward(self) -> None:
        torch.manual_seed(3)
        model = HRM(tiny_config("h", h_cycles=1, l_cycles=6))
        input_ids = torch.randint(0, 11, (2, 4))
        native_carry, native_logits = model(model.initial_carry, input_ids)
        embedding = model.embed(input_ids)
        state = initial_hrm_state(model, input_ids)
        phase = 0
        updates = 0
        for _ in range(6):
            state, phase, updated_h = advance_hrm_l(model, state, embedding, phase, 6)
            updates += int(updated_h)
        self.assertEqual((phase, updates), (0, 1))
        self.assertTrue(torch.equal(state[0], native_carry["z_H"]))
        self.assertTrue(torch.equal(state[1], native_carry["z_L"]))
        with torch.inference_mode():
            manual_logits = model.readout_logits(*state)
        self.assertTrue(torch.equal(manual_logits, native_logits))


if __name__ == "__main__":
    unittest.main()
