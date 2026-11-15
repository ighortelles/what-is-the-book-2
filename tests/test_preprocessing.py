"""CPU tests for image contracts and restart behavior, without model downloads."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.detection import train_preprocessing as runner
from src.preprocess.image_variants import (
    dataset_identity,
    prepare_variant,
    read_image,
    release_variant,
    transform,
)

CONFIG = json.loads((runner.ROOT / "configs/yolov8_preprocessing.json").read_text())


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data/unified"
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.image[8:40, 20:44] = [40, 180, 250]
        self.label = b"0 0.5 0.5 0.25 0.5\n"
        entries = []
        for i, split in enumerate(["train", "val"]):
            images = self.data / split / "images"
            labels = self.data / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            cv2.imencode(".png", self.image)[1].tofile(str(images / "sample.png"))
            (labels / "sample.txt").write_bytes(self.label)
            entries.append(
                {
                    "id": i,
                    "group": i,
                    "source": "fixture",
                    "split": split,
                    "image": f"{split}/images/sample.png",
                }
            )
        # Test image intentionally missing: it must never be opened by preprocessing.
        entries.append(
            {
                "id": 2,
                "group": 2,
                "source": "fixture",
                "split": "test",
                "image": "test/images/reserved.png",
            }
        )
        (self.data / "manifest.json").write_text(json.dumps(entries))

    def test_transform_contracts(self):
        for variant in CONFIG["variants"]:
            with self.subTest(variant=variant["name"]):
                before = self.image.copy()
                output = transform(self.image, variant)
                np.testing.assert_array_equal(self.image, before)
                np.testing.assert_array_equal(output, transform(self.image, variant))
                self.assertEqual(output.shape, before.shape)
                self.assertEqual(output.dtype, np.uint8)
                if variant["operator"] == "original":
                    np.testing.assert_array_equal(output, before)
                if variant["operator"] in {"canny", "sobel", "laplacian"}:
                    self.assertTrue(output.any())
                    self.assertFalse(transform(np.zeros_like(before), variant).any())
                    np.testing.assert_array_equal(output[:, :, 0], output[:, :, 1])
                if variant["operator"] == "canny":
                    self.assertTrue(set(np.unique(output)) <= {0, 255})

    def test_export_identity_and_cleanup(self):
        entries, fingerprint = dataset_identity(self.data)
        self.assertEqual(len(entries), 2)
        for variant in CONFIG["variants"]:
            destination = self.root / "cache" / variant["name"]
            signature = {"variant": variant, "fingerprint": fingerprint}
            prepare_variant(self.data, destination, entries, variant, signature)
            self.assertFalse((destination / "test").exists())
            self.assertEqual((destination / "val/labels/sample.txt").read_bytes(), self.label)
            np.testing.assert_array_equal(
                read_image(destination / "val/images/sample.png"), transform(self.image, variant)
            )
            with self.assertRaises(RuntimeError):
                prepare_variant(self.data, destination, entries, variant, {"different": True})
            release_variant(destination, self.root / "cache", signature)
            self.assertFalse(destination.exists())
        label = self.data / "train/labels/sample.txt"
        label.write_bytes(self.label + self.label)
        self.assertNotEqual(dataset_identity(self.data)[1], fingerprint)

    def test_runner_start_skip_resume_and_finalize(self):
        config = copy.deepcopy(CONFIG)
        config["variants"] = config["variants"][:2]
        (self.root / "configs").mkdir()
        (self.root / "configs/yolov8_preprocessing.json").write_text(json.dumps(config))
        implementation = self.root / "src/preprocess/image_variants.py"
        implementation.parent.mkdir(parents=True)
        implementation.write_text("fixture")
        calls = []

        class Model:
            def __init__(self, path):
                self.path = path

            def train(self, **kwargs):
                calls.append(kwargs)
                folder = (
                    Path(self.path).parent.parent
                    if kwargs.get("resume")
                    else Path(kwargs["project"]) / kwargs["name"]
                )
                (folder / "weights").mkdir(parents=True, exist_ok=True)
                (folder / "weights/best.pt").write_text("fake checkpoint")
                (folder / "weights/last.pt").write_text("fake checkpoint")

            def val(self, **kwargs):
                assert kwargs["split"] == "val"
                return SimpleNamespace(results_dict={"metrics/mAP50-95(B)": 0.5})

        torch = SimpleNamespace(
            __version__="test",
            cuda=SimpleNamespace(is_available=lambda: True),
            load=lambda *a, **kw: {"epoch": 38, "optimizer": {}},
        )
        ultra = SimpleNamespace(__version__="test", YOLO=Model)
        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner.importlib.util, "find_spec", return_value=None),
            patch.dict(sys.modules, {"torch": torch, "ultralytics": ultra}),
            patch.object(sys, "argv", ["runner", "--run"]),
        ):
            runner.main()
            self.assertEqual(len(calls), 2)
            runner.main()
            self.assertEqual(len(calls), 2)
            result = self.root / "runs/preprocessing/grid/original/result.json"
            result.unlink()
            runner.main()
            self.assertEqual(len(calls), 3)
            self.assertTrue(calls[-1]["resume"])
            result.unlink()
            torch.load = lambda *a, **kw: {"epoch": -1, "optimizer": None}
            runner.main()
            self.assertEqual(len(calls), 3)
            self.assertTrue(result.exists())


if __name__ == "__main__":
    unittest.main()
