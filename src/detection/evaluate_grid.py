"""Evaluate completed original YOLOv8 experiments on the held-out test set."""

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path


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
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("Batch must be positive")
    data, grid, output = args.data.resolve(), args.grid.resolve(), args.output.resolve()
    if output == grid or output.is_relative_to(data):
        raise ValueError("Use a separate output directory outside the dataset")
    manifest_hash, test_hash, count = inspect_data(data)
    experiments = []
    for number in range(1, 9):
        folder = grid / f"exp_{number:02d}"
        result_path, best = folder / "result.json", folder / "weights/best.pt"
        if not result_path.is_file() or not best.is_file():
            raise FileNotFoundError(f"{folder.name}: finish training and validation first")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["dataset_sha256"] != manifest_hash:
            raise ValueError(f"{folder.name}: dataset differs from the training manifest")
        experiments.append((folder.name, result, best))
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
    for name, result, best in experiments:
        destination = output / name
        destination.mkdir(exist_ok=True)
        summary = destination / "test_metrics.json"
        if summary.exists():
            saved = json.loads(summary.read_text(encoding="utf-8"))
            print(f"STATUS | {name}: test already evaluated; skipping", flush=True)
        else:
            print(f"STATUS | {name}: evaluating {count} test images", flush=True)
            metrics = ultra.YOLO(str(best)).val(
                data=str(yaml),
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
        rows.append(
            {
                "experiment": name,
                "selected_by_validation": name == selected,
                "imgsz": result["settings"]["imgsz"],
                "lr0": result["settings"]["lr0"],
                "mosaic": result["settings"]["mosaic"],
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
    print(f"STATUS | Finished. Model selected on validation: {selected}. Reports: {output}")


if __name__ == "__main__":
    main()
