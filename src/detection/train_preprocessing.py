"""Train/resume controlled YOLOv8 preprocessing comparisons without evaluating test."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.preprocess.image_variants import (
    dataset_identity,
    prepare_variant,
    preview,
    release_variant,
)

ROOT = Path(__file__).resolve().parents[2]


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_ranking(output: Path) -> None:
    results = [json.loads(p.read_text()) for p in output.glob("*/result.json")]
    results.sort(key=lambda r: r["metrics"]["metrics/mAP50-95(B)"], reverse=True)
    save_json(output / "ranking.json", results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", help="Optional single variant; otherwise run all")
    parser.add_argument("--data", type=Path, default=ROOT / "data/unified")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/preprocessed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/preprocessing")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/yolov8_preprocessing.json").read_text())
    variants = cfg["variants"]
    if args.variant:
        variants = [v for v in variants if v["name"] == args.variant]
        if not variants:
            parser.error("Unknown variant")
    if args.preview:
        preview(args.data, variants, args.output / "previews")
        print(f"STATUS | Previews: {args.output / 'previews'}", flush=True)
    if not args.run:
        print(json.dumps({**cfg, "variants": variants}, indent=2))
        return
    os.environ["MPLBACKEND"] = "Agg"
    # Ultralytics optionally enables blur/CLAHE via Albumentations if installed.
    # Do not let that silently change an experiment testing image filters.
    if importlib.util.find_spec("albumentations") is not None:
        raise RuntimeError("Use the isolated locked environment without Albumentations")
    torch = importlib.import_module("torch")
    ultra = importlib.import_module("ultralytics")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU in Colab")
    entries, content_hash = dataset_identity(args.data)
    implementation = ROOT / "src/preprocess/image_variants.py"
    code_hash = hashlib.sha256(implementation.read_bytes()).hexdigest()
    output = args.output.resolve() / ("smoke" if args.smoke else "grid")
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve()
    if cache == args.data.resolve() or cache.is_relative_to(args.data.resolve()):
        raise ValueError("Generated cache must be outside the original dataset")
    settings = dict(cfg["fixed"])
    if args.smoke:
        settings.update(epochs=1, fraction=0.05, plots=False)
    for variant in variants:
        name = variant["name"]
        folder = output / name
        result_path = folder / "result.json"
        request_path = output / f"{name}_request.json"
        signature = {
            "variant": variant,
            "model": cfg["model"],
            "settings": settings,
            "train_val_sha256": content_hash,
            "preprocessing_sha256": code_hash,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ultralytics": ultra.__version__,
            "schema": 1,
        }
        if result_path.exists():
            if json.loads(result_path.read_text())["request"] != signature:
                raise RuntimeError(f"{name}: incompatible completed experiment; use new output")
            print(f"STATUS | {name}: completed; skipping", flush=True)
            continue
        if request_path.exists():
            if json.loads(request_path.read_text()) != signature:
                raise RuntimeError(f"{name}: configuration/data/version changed")
        else:
            if folder.exists():
                raise RuntimeError(f"{name}: existing experiment without provenance")
            save_json(request_path, signature)
        generated = cache / name
        print(f"STATUS | {name}: preparing train and validation PNGs", flush=True)
        prepare_variant(args.data, generated, entries, variant, signature)
        # This YAML intentionally does not expose the held-out test set.
        yaml = output / f"{name}_dataset.yaml"
        yaml.write_text(
            f"path: {generated.as_posix()}\ntrain: train/images\nval: val/images\n"
            "names: [book]\nnc: 1\n",
            encoding="utf-8",
        )
        last, best = folder / "weights/last.pt", folder / "weights/best.pt"
        if last.exists():
            # Only load trusted checkpoints created by these experiments.
            checkpoint = torch.load(last, map_location="cpu", weights_only=False)
            epoch = int(checkpoint.get("epoch", -1))
            can_resume = (
                epoch >= 0
                and checkpoint.get("optimizer") is not None
                and epoch + 1 < settings["epochs"]
            )
            finalized = epoch == -1 or epoch + 1 >= settings["epochs"]
            del checkpoint
            if can_resume:
                print(f"STATUS | {name}: resuming after epoch {epoch + 1}", flush=True)
                ultra.YOLO(str(last)).train(resume=True, data=str(yaml), device=args.device)
            elif not finalized or not best.exists():
                raise RuntimeError(f"{name}: checkpoint cannot be safely resumed/finalized")
        elif folder.exists():
            raise RuntimeError(f"{name}: directory exists without last.pt; inspect its log")
        else:
            print(f"STATUS | {name}: training", flush=True)
            ultra.YOLO(cfg["model"]).train(
                data=str(yaml),
                device=args.device,
                project=str(output),
                name=name,
                exist_ok=False,
                **settings,
            )
        if not best.is_file():
            raise RuntimeError(f"{name}: missing best.pt")
        print(f"STATUS | {name}: evaluating best.pt on validation", flush=True)
        metrics = ultra.YOLO(str(best)).val(
            data=str(yaml),
            split="val",
            imgsz=settings["imgsz"],
            batch=settings["batch"],
            device=args.device,
            max_det=1000,
            project=str(folder),
            name="validation",
            plots=True,
        )
        save_json(
            result_path,
            {
                "request": signature,
                "best": str(best),
                "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
            },
        )
        update_ranking(output)
        if not args.keep_data:
            release_variant(generated, cache, signature)
        print(f"STATUS | {name}: completed and saved", flush=True)
    update_ranking(output)
    print(f"STATUS | Finished. Ranking: {output / 'ranking.json'}", flush=True)


if __name__ == "__main__":
    main()
