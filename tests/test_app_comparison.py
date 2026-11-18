"""Select the best unfiltered and filtered models without confusing the control."""

import json
import tempfile
import unittest
from pathlib import Path

from src.app.server import comparison_models, discover


class ComparisonTests(unittest.TestCase):
    def test_portable_models_keep_preprocessing_and_test_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            for name, variant, size in (
                ("exp_05", {"name": "original", "operator": "original"}, 960),
                (
                    "preproc__bilateral_5",
                    {
                        "name": "bilateral_5",
                        "operator": "bilateral",
                        "d": 5,
                        "sigma_color": 50,
                        "sigma_space": 50,
                    },
                    640,
                ),
            ):
                folder = storage / "models" / name
                folder.mkdir(parents=True)
                (folder / "best.pt").touch()
                (folder / "result.json").write_text(
                    json.dumps(
                        {
                            "request": {"settings": {"imgsz": size}, "variant": variant},
                            "metrics": {"metrics/mAP50-95(B)": 0.8},
                        }
                    )
                )
            (storage / "models/ranking.json").write_text(
                json.dumps(
                    [
                        {"experiment": "exp_05", "test_mAP50_95": 0.77},
                        {"experiment": "preproc__bilateral_5", "test_mAP50_95": 0.66},
                    ]
                )
            )
            models = discover(storage)
            selected = comparison_models(storage, models)
            self.assertEqual([x["basis"] for x in selected], ["teste", "teste"])
            self.assertEqual([models[x["id"]]["imgsz"] for x in selected], [960, 640])
            self.assertEqual(models[selected[1]["id"]]["variant"]["sigma_color"], 50)

    def test_ranking_and_validation_fallback(self) -> None:
        models = {}
        for name, operator, score in (
            ("exp_05", "original", 0.9),
            ("preproc__original", "original", 0.8),
            ("preproc__bilateral", "bilateral", 0.7),
            ("preproc__gaussian", "gaussian", 0.75),
        ):
            models[name] = {
                "experiment": name,
                "label": name,
                "validation_map": score,
                "variant": {"name": name, "operator": operator},
            }
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            fallback = comparison_models(storage, models)
            self.assertEqual(fallback[1]["id"], "preproc__gaussian")
            self.assertEqual(fallback[1]["basis"], "validação")
            ranking = storage / "runs/final_test_all/ranking.json"
            ranking.parent.mkdir(parents=True)
            ranking.write_text(
                json.dumps(
                    [
                        {"experiment": name, "test_mAP50_95": score}
                        for name, score in (
                            ("exp_05", 0.77),
                            ("preproc__original", 0.67),
                            ("preproc__bilateral", 0.66),
                            ("preproc__gaussian", 0.65),
                        )
                    ]
                )
            )
            selection = comparison_models(storage, models)
            self.assertEqual([item["id"] for item in selection], ["exp_05", "preproc__bilateral"])
            self.assertTrue(all(item["basis"] == "teste" for item in selection))
            with self.assertRaises(ValueError):
                comparison_models(storage, {"exp_05": models["exp_05"]})
