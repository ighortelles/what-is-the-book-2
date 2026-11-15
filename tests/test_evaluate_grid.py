"""Check test isolation, validation selection and safe evaluation reuse."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.detection import evaluate_grid as evaluator


class EvaluationTests(unittest.TestCase):
    def test_evaluate_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, grid, output = root / "data", root / "grid", root / "evaluation"
            (data / "test/images").mkdir(parents=True)
            (data / "test/labels").mkdir()
            (data / "test/images/example.jpg").write_bytes(b"mock image")
            (data / "test/labels/example.txt").write_text("0 0.5 0.5 0.2 0.2")
            manifest = data / "manifest.json"
            entries = [{"group": "one", "split": "test", "image": "test/images/example.jpg"}]
            manifest.write_text(json.dumps(entries))
            metrics = {
                "metrics/mAP50-95(B)": 0.6,
                "metrics/mAP50(B)": 0.8,
                "metrics/precision(B)": 0.7,
                "metrics/recall(B)": 0.9,
            }
            for number in range(1, 9):
                folder = grid / f"exp_{number:02d}"
                (folder / "weights").mkdir(parents=True)
                (folder / "weights/best.pt").write_bytes(b"trusted mock checkpoint")
                result = {
                    "dataset_sha256": evaluator.digest(manifest),
                    "ultralytics": "test",
                    "settings": {"imgsz": 640, "lr0": 0.001, "mosaic": 0},
                    "metrics": {"metrics/mAP50-95(B)": 0.95 if number == 3 else 0.5},
                }
                (folder / "result.json").write_text(json.dumps(result))
            (grid / "exp_07/result.json").unlink()
            (grid / "exp_08/weights/best.pt").unlink()
            calls = []

            def validate(**kwargs):
                calls.append(kwargs)
                self.assertTrue((output / "evaluation_plan.json").is_file())
                return SimpleNamespace(results_dict=metrics, speed={"inference": 1.0})

            ultra = SimpleNamespace(
                __version__="test", YOLO=lambda _: SimpleNamespace(val=validate)
            )
            torch = SimpleNamespace(__version__="test")
            argv = ["evaluate", "--data", str(data), "--grid", str(grid), "--output", str(output)]
            with (
                patch("sys.argv", argv),
                patch.object(evaluator.importlib, "import_module", side_effect=[ultra, torch] * 2),
            ):
                evaluator.main()
                evaluator.main()
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(call["split"] == "test" for call in calls))
            plan = json.loads((output / "evaluation_plan.json").read_text())
            self.assertEqual(plan["selected_by_validation"], "exp_03")
            self.assertEqual(len(json.loads((output / "comparison.json").read_text())), 6)
            self.assertEqual(set(plan["skipped_experiments"]), {"exp_07", "exp_08"})
            for checkpoint in grid.glob("*/weights/best.pt"):
                checkpoint.unlink()
            with patch("sys.argv", argv), self.assertRaisesRegex(RuntimeError, "No completed"):
                evaluator.main()
            entries.append({"group": "one", "split": "train", "image": "train/images/x.jpg"})
            manifest.write_text(json.dumps(entries))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                evaluator.inspect_data(data)


if __name__ == "__main__":
    unittest.main()
