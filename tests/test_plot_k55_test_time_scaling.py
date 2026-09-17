import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_k55_test_time_scaling import (
    CONDITIONS,
    TEST_BANDS,
    TRAIN_BANDS,
    MODELS,
    load_summary,
    l6_bar_data,
    l6_table_rows,
    points_from_workers,
)


class PlotK55TestTimeScalingTests(unittest.TestCase):
    def test_loads_complete_three_seed_summary(self):
        fields = [
            "model", "train_band", "L_cycles", "num_seeds",
            "easy_exact_match_mean", "easy_exact_match_std",
            "hard_exact_match_mean", "hard_exact_match_std",
        ]
        rows = [
            {
                "model": model, "train_band": train_band, "L_cycles": 6, "num_seeds": 3,
                "easy_exact_match_mean": 0.8, "easy_exact_match_std": 0.01,
                "hard_exact_match_mean": 0.6, "hard_exact_match_std": 0.02,
            }
            for model in MODELS for train_band in TRAIN_BANDS
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            points = load_summary(path)
        self.assertEqual(len(points), len(MODELS) * len(TRAIN_BANDS) * len(TEST_BANDS))
        self.assertEqual({point.test_band for point in points}, set(TEST_BANDS))
        self.assertEqual(l6_table_rows(points)[0], ["Easy train", "Easy test", "80.00 +/- 1.00", "80.00 +/- 1.00", "80.00 +/- 1.00"])
        labels, grouped = l6_bar_data(points)
        self.assertEqual(labels, ["Easy -> Easy", "Easy -> Hard", "Hard -> Easy", "Hard -> Hard"])
        self.assertEqual([point.mean for point in grouped["hrm"]], [80.0, 60.0, 80.0, 60.0])

    def test_recomputes_a_condition_after_excluding_one_seed(self):
        fields = [
            "condition", "model", "train_band", "seed", "L_cycles", "test_band",
            "total_samples", "exact_match_accuracy", "checkpoint",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            workers_dir = Path(temp_dir)
            for condition, (model, train_band) in CONDITIONS.items():
                for seed in (1, 2, 3):
                    worker_dir = workers_dir / f"{condition}_seed{seed}"
                    worker_dir.mkdir()
                    rows = [
                        {
                            "condition": condition, "model": model, "train_band": train_band,
                            "seed": seed, "L_cycles": 6, "test_band": test_band,
                            "total_samples": 10_000,
                            "exact_match_accuracy": seed / 10,
                            "checkpoint": "checkpoint.pt",
                        }
                        for test_band in TEST_BANDS
                    ]
                    with (worker_dir / "test_time_scaling_per_seed.csv").open("w", newline="") as output:
                        writer = csv.DictWriter(output, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
            points = points_from_workers(workers_dir, {("easy_k55_hrm", 2)})
        easy_hrm_hard = next(
            point for point in points
            if point.model == "hrm" and point.train_band == "easy" and point.test_band == "hard"
        )
        self.assertEqual(easy_hrm_hard.num_seeds, 2)
        self.assertAlmostEqual(easy_hrm_hard.mean, 20.0)


if __name__ == "__main__":
    unittest.main()
