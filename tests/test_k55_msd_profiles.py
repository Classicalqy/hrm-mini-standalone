import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

coolname = types.ModuleType("coolname")
coolname.generate_slug = lambda _count: "stub"
sys.modules.setdefault("coolname", coolname)

from arch.trm import TRM
from scripts.analyze_long_rollout_msd import RunDirectory
from scripts.core_five_l_depth_long_rollout import advance_hrm_l, initial_hrm_state
from scripts.k55_msd_profiles import (
    discover_k55_runs,
    inferred_k55_config,
    k55_units,
    last_k55_checkpoints,
    load_k55_config,
)
from test_hrm_readout import tiny_config


class K55MSDProfilesTest(unittest.TestCase):
    def test_legacy_omegacon_metadata_uses_verified_condition_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            seed_dir = Path(temporary)
            (seed_dir / "model_config.json").write_text(
                "data: !!python/object:omegaconf.dictconfig.DictConfig\n"
                "  _parent: &parent !!python/object:omegaconf.dictconfig.DictConfig\n"
                "    _parent: *parent\n"
            )
            config = load_k55_config(seed_dir, "hard_k55_trm")
        self.assertEqual(config.arch.name, "trm@TRM")
        self.assertEqual((config.arch.__pydantic_extra__ or {})["L_cycles"], 6)
        self.assertEqual((config.data.__pydantic_extra__ or {})["blank_min"], 56)

    def test_inferred_configs_match_the_four_checkpoint_architectures(self) -> None:
        expected = {
            "easy_k55_hrm": ("hrm@HRM", 2, 6),
            "easy_k55_trm": ("trm@TRM", 2, 6),
            "easy_k55_hrm_h2l1": ("hrm@HRM", 2, 1),
            "easy_k55_rt": ("rt@RecurrentTransformer", None, None),
        }
        for condition, (name, h_cycles, l_cycles) in expected.items():
            with self.subTest(condition=condition):
                config = inferred_k55_config(condition)
                arch = config.arch.__pydantic_extra__ or {}
                self.assertEqual(config.arch.name, name)
                self.assertEqual(arch.get("H_cycles"), h_cycles)
                self.assertEqual(arch.get("L_cycles"), l_cycles)
                self.assertEqual(config.data.name, "sudoku")

    def test_trm_trace_and_manual_l_updates_match_native_forward(self) -> None:
        torch.manual_seed(11)
        config = tiny_config("h", h_cycles=2, l_cycles=3)
        model = TRM(config)
        x = torch.randint(0, 11, (2, 4))
        native, native_logits = model(model.initial_carry, x)
        events: list[str] = []
        traced, traced_logits = model.forward_with_trace(
            model.initial_carry, x, lambda event, _h, _l: events.append(event),
        )
        self.assertEqual(events, ["l", "l", "l", "h", "l", "l", "l", "h"])
        self.assertTrue(torch.equal(native_logits, traced_logits))
        self.assertTrue(torch.equal(native["z_H"], traced["z_H"]))
        self.assertTrue(torch.equal(native["z_L"], traced["z_L"]))

        state = initial_hrm_state(model, x)
        phase = 0
        embedding = model.embed(x)
        for _ in range(6):
            state, phase, _ = advance_hrm_l(model, state, embedding, phase, 3)
        self.assertEqual(phase, 0)
        self.assertTrue(torch.equal(native["z_H"], state[0]))
        self.assertTrue(torch.equal(native["z_L"], state[1]))

    def test_k55_units_sweep_hrm_and_trm_only(self) -> None:
        runs = []
        for difficulty in ("easy", "hard"):
            for model_name in ("hrm", "trm", "hrm_h2l1", "rt"):
                kind = "rt" if model_name == "rt" else ("trm" if model_name == "trm" else "hrm")
                runs.append(RunDirectory(kind, f"{difficulty}_k55_{model_name}", 1, None, None,
                                         None if kind == "rt" else (1 if model_name == "hrm_h2l1" else 6),
                                         "rt" if kind == "rt" else "h"))  # type: ignore[arg-type]
        units = k55_units(runs, (6, 8, 16))
        self.assertEqual(len(units), 16)
        self.assertEqual({unit.eval_l for unit in units if unit.run.condition.endswith("_hrm")}, {6, 8, 16})
        self.assertEqual({unit.eval_l for unit in units if unit.run.condition.endswith("_trm")}, {6, 8, 16})
        self.assertEqual({unit.eval_l for unit in units if unit.run.condition.endswith("_rt")}, {1})

    def test_last_checkpoint_policy_uses_highest_epoch_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            seed_dir = Path(temporary) / "easy_k55_hrm" / "seed_1"
            seed_dir.mkdir(parents=True)
            for epoch in (0, 7, 19):
                (seed_dir / f"epoch_{epoch}.pt").touch()
            run = RunDirectory(
                "hrm", "easy_k55_hrm", 1, seed_dir, None, 6, "h",  # type: ignore[arg-type]
            )
            selected = last_k55_checkpoints([run])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["epoch"], 19)
        self.assertEqual(Path(str(selected[0]["checkpoint"])).name, "epoch_19.pt")
        self.assertEqual(selected[0]["selection_policy"], "last")
        self.assertEqual(selected[0]["test_exact_match"], "")

    def test_discovery_accepts_exact_requested_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = {}
            for difficulty in ("easy", "hard"):
                for model_name in ("hrm", "trm", "rt", "hrm_h2l1"):
                    seed_dir = root / f"{difficulty}_k55_{model_name}" / "seed_1"
                    seed_dir.mkdir(parents=True)
                    (seed_dir / "epoch_0.pt").touch()
                    arch_name = {"hrm": "hrm@HRM", "hrm_h2l1": "hrm@HRM", "trm": "trm@TRM", "rt": "rt@RecurrentTransformer"}[model_name]
                    l_cycles = 1 if model_name == "hrm_h2l1" else 6
                    configs[seed_dir] = SimpleNamespace(
                        arch=SimpleNamespace(name=arch_name, __pydantic_extra__={} if model_name == "rt" else {"H_cycles": 2, "L_cycles": l_cycles}),
                    )
            with patch("scripts.k55_msd_profiles.load_config", side_effect=lambda path: configs[path]):
                runs = discover_k55_runs(root, (1,))
        self.assertEqual(len(runs), 8)
        self.assertEqual(sum(run.kind == "trm" for run in runs), 2)


if __name__ == "__main__":
    unittest.main()
