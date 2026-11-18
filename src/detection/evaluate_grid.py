"""Evaluate original and preprocessing YOLOv8 experiments on the held-out test set."""

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import image_variants


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def inspect_data(data: Path) -> tuple[str, str, int]:
    manifest_path = data / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: dict[str, str] = {}
    test = []
    for entry in entries:
        group = str(entry["group"])
        split = entry["split"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Group {group} appears in multiple splits")
        groups[group] = split
        if split == "test":
            test.append(entry)
    if not test:
        raise ValueError("Test split is empty")
    signature = hashlib.sha256()
    expected = set()
    for entry in sorted(test, key=lambda item: item["image"]):
        image = (data / entry["image"]).resolve()
        if not image.is_relative_to((data / "test/images").resolve()):
            raise ValueError("Invalid test image path")
        label = data / "test/labels" / (image.stem + ".txt")
        expected.add(image)
        signature.update(entry["image"].encode())
        signature.update(bytes.fromhex(digest(image)))
        signature.update(bytes.fromhex(digest(label)))
    actual = {p.resolve() for p in (data / "test/images").iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError("Test image directory differs from the manifest")
    return digest(manifest_path), signature.hexdigest(), len(test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--grid", type=Path, action="append", default=[])
    parser.add_argument("--preprocessing-grid", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("Batch must be positive")
    data, output = args.data.resolve(), args.output.resolve()
    grids = [path.resolve() for path in args.grid]
    preprocessing = args.preprocessing_grid
    if not grids and preprocessing is None:
        parser.error("Provide --grid and/or --preprocessing-grid")
    if output in grids or output.is_relative_to(data):
        raise ValueError("Use a separate output directory outside the dataset")
    manifest_hash, test_hash, count = inspect_data(data)
    experiments = []
    skipped = {}
    for number in range(1, 9) if grids else []:
        name = f"exp_{number:02d}"
        candidates = []
        for grid in grids:
            folder = grid / name
            result_path = folder / "result.json"
            best = folder / "weights/best.pt"
            if not best.is_file():
                best = folder / "best.pt"  # Compact exports downloaded from Drive.
            if result_path.is_file() and best.is_file():
                candidates.append((result_path, best))
        if not candidates:
            skipped[name] = "Missing result.json or best.pt in configured original grids"
            print(f"STATUS | {name}: incomplete or unavailable; skipping", flush=True)
            continue
        result_path, best = candidates[0]
        if any(digest(other) != digest(best) for _, other in candidates[1:]):
            raise ValueError(f"{name}: conflicting checkpoints in configured grids")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["dataset_sha256"] != manifest_hash:
            raise ValueError(f"{name}: dataset differs from the training manifest")
        result["family"] = "original_grid"
        experiments.append((name, result, best))
    if preprocessing is not None:
        if not preprocessing.is_dir():
            raise FileNotFoundError(f"Preprocessing grid not found: {preprocessing}")
        _, train_val_hash = image_variants.dataset_identity(data)
        implementation_hash = digest(Path(image_variants.__file__))
        for folder in sorted(preprocessing.iterdir()):
            if not folder.is_dir():
                continue
            name = f"preproc__{folder.name}"
            result_path, best = folder / "result.json", folder / "weights/best.pt"
            if not result_path.is_file() or not best.is_file():
                skipped[name] = "Missing result.json or weights/best.pt"
                print(f"STATUS | {name}: incomplete; skipping", flush=True)
                continue
            saved_result = json.loads(result_path.read_text(encoding="utf-8"))
            request = saved_result["request"]
            if request["train_val_sha256"] != train_val_hash:
                raise ValueError(f"{name}: train/validation dataset changed")
            if request["preprocessing_sha256"] != implementation_hash:
                raise ValueError(f"{name}: preprocessing implementation changed")
            if request["opencv"] != cv2.__version__ or request["numpy"] != np.__version__:
                raise ValueError(f"{name}: install the OpenCV/NumPy versions used in training")
            result = {**request, "metrics": saved_result["metrics"], "family": "preprocessing"}
            experiments.append((name, result, best))
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "skipped_experiments.json", skipped)
    if not experiments:
        raise RuntimeError("No completed experiments available; nothing to evaluate")
    print(f"STATUS | Evaluating {len(experiments)} completed experiments", flush=True)
    selected = max(experiments, key=lambda e: e[1]["metrics"]["metrics/mAP50-95(B)"])[0]
    os.environ["MPLBACKEND"] = "Agg"
    ultra = importlib.import_module("ultralytics")
    torch = importlib.import_module("torch")
    for name, result, _ in experiments:
        if result["ultralytics"] != ultra.__version__:
            raise ValueError(f"{name}: install ultralytics=={result['ultralytics']}")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "selected_by_validation": selected,
        "skipped_experiments": skipped,
        "manifest_sha256": manifest_hash,
        "test_sha256": test_hash,
        "test_images": count,
        "ultralytics": ultra.__version__,
        "torch": torch.__version__,
        "device": args.device,
        "batch": args.batch,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 1000,
        "half": False,
        "augment": False,
        "checkpoints": {name: digest(best) for name, _, best in experiments},
        "settings": {name: result["settings"] for name, result, _ in experiments},
        "variants": {name: result.get("variant") for name, result, _ in experiments},
        "validation_metrics": {name: result["metrics"] for name, result, _ in experiments},
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "preprocessing_sha256": digest(Path(image_variants.__file__)),
        "evaluator_sha256": digest(Path(__file__)),
    }
    plan_path = output / "evaluation_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Evaluation configuration changed; choose a new output directory")
    save_json(plan_path, plan)  # Selection is frozen before any test inference.
    yaml = output / "test_dataset.yaml"
    yaml.write_text(
        f"path: {json.dumps(data.as_posix())}\ntrain: train/images\nval: val/images\n"
        "test: test/images\nnames: [book]\nnc: 1\n",
        encoding="utf-8",
    )
    rows = []
    cache = (args.cache or output / "generated_test").resolve()
    if cache == data or cache.is_relative_to(data):
        raise ValueError("Generated test cache must be outside the source dataset")
    test_entries = [
        item
        for item in json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        if item["split"] == "test"
    ]
    for name, result, best in experiments:
        destination = output / name
        destination.mkdir(exist_ok=True)
        summary = destination / "test_metrics.json"
        if summary.exists():
            saved = json.loads(summary.read_text(encoding="utf-8"))
            print(f"STATUS | {name}: test already evaluated; skipping", flush=True)
        else:
            print(f"STATUS | {name}: evaluating {count} test images", flush=True)
            evaluation_yaml = yaml
            generated = None
            if "variant" in result:
                generated = cache / name
                image_variants.prepare_variant(
                    data, generated, test_entries, result["variant"], plan
                )
                evaluation_yaml = destination / "test_dataset.yaml"
                evaluation_yaml.write_text(
                    f"path: {json.dumps(generated.as_posix())}\n"
                    "train: test/images\nval: test/images\ntest: test/images\n"
                    "names: [book]\nnc: 1\n",
                    encoding="utf-8",
                )
            metrics = ultra.YOLO(str(best)).val(
                data=str(evaluation_yaml),
                split="test",
                imgsz=result["settings"]["imgsz"],
                batch=args.batch,
                device=args.device,
                workers=0,
                conf=0.001,
                iou=0.7,
                max_det=1000,
                half=False,
                augment=False,
                plots=True,
                project=str(destination),
                name="plots",
                exist_ok=True,
                verbose=False,
            )
            saved = {
                "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
                "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
            }
            save_json(summary, saved)
            if generated is not None:
                image_variants.release_variant(generated, cache, plan)
        rows.append(
            {
                "experiment": name,
                "family": result["family"],
                "preprocessing": result.get("variant", {}).get("name", "none"),
                "selected_by_validation": name == selected,
                "imgsz": result["settings"]["imgsz"],
                "lr0": result["settings"]["lr0"],
                "mosaic": result["settings"]["mosaic"],
                "training_batch": result["settings"].get("batch"),
                "val_mAP50_95": result["metrics"]["metrics/mAP50-95(B)"],
                "test_precision": saved["metrics"]["metrics/precision(B)"],
                "test_recall": saved["metrics"]["metrics/recall(B)"],
                "test_mAP50": saved["metrics"]["metrics/mAP50(B)"],
                "test_mAP50_95": saved["metrics"]["metrics/mAP50-95(B)"],
            }
        )
        save_json(output / "comparison.json", rows)
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: (-row["test_mAP50_95"], row["experiment"]))
    ranking = [{"rank": i, **row} for i, row in enumerate(ranked, 1)]
    save_json(output / "ranking.json", ranking)
    with (output / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    for row in ranking:
        print(
            f"STATUS | Rank {row['rank']}: {row['experiment']} "
            f"test mAP50-95={row['test_mAP50_95']:.4f}",
            flush=True,
        )
    print(f"STATUS | Finished. Model selected on validation: {selected}. Reports: {output}")


if __name__ == "__main__":
    main()
