"""Preview or execute a fixed YOLOv8 grid, selecting only on validation."""

import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute; otherwise print the plan")
    parser.add_argument("--experiment", type=int, help="Run only this experiment (1 to 8)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--smoke", action="store_true", help="One epoch, 5%% of train; not a result"
    )
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/yolov8_grid.json").read_text())
    keys = list(cfg["grid"])
    combinations = list(itertools.product(*(cfg["grid"][key] for key in keys)))
    if args.experiment is not None and not 1 <= args.experiment <= len(combinations):
        parser.error("Experiment is outside the configured grid")
    plan = []
    for i, values in enumerate(combinations, 1):
        if args.experiment is not None and args.experiment != i:
            continue
        settings = {**cfg["fixed"], **dict(zip(keys, values, strict=True))}
        if args.smoke:
            settings.update(epochs=1, fraction=0.05, close_mosaic=0, plots=False)
        plan.append({"id": i, "model": cfg["model"], "settings": settings})
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    # Colab's inline backend is inherited by subprocesses but is not installed
    # in the isolated training environment. Save plots without an interactive UI.
    os.environ["MPLBACKEND"] = "Agg"
    ultralytics = importlib.import_module("ultralytics")
    torch = importlib.import_module("torch")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Colab, or explicitly select --device cpu")
    data = ROOT / "data/unified"
    manifest_bytes = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert {m["split"] for m in manifest} == {"train", "val", "test"}
    # Portable configuration, regenerated on the execution machine.
    output = args.output.resolve() / ("smoke" if args.smoke else "grid")
    output.mkdir(parents=True, exist_ok=True)
    yaml = output / "dataset.yaml"
    yaml.write_text(
        f"path: {data.as_posix()}\ntrain: train/images\nval: val/images\n"
        "test: test/images\nnames: [book]\nnc: 1\n",
        encoding="utf-8",
    )
    for experiment in plan:
        name = f"exp_{experiment['id']:02d}"
        destination = output / name
        if destination.exists():
            raise RuntimeError(f"{destination} exists; refusing to overwrite an experiment")
        request = {
            **experiment,
            "dataset_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "device": args.device,
        }
        (output / f"{name}_request.json").write_text(json.dumps(request, indent=2))
        model = ultralytics.YOLO(cfg["model"])
        model.train(
            data=str(yaml),
            device=args.device,
            project=str(output),
            name=name,
            exist_ok=False,
            **experiment["settings"],
        )
        best = destination / "weights/best.pt"
        metrics = ultralytics.YOLO(str(best)).val(
            data=str(yaml),
            split="val",
            imgsz=experiment["settings"]["imgsz"],
            batch=experiment["settings"]["batch"],
            device=args.device,
            max_det=1000,
            project=str(destination),
            name="validation",
            plots=True,
        )
        result = {**request, "metrics": metrics.results_dict, "best": str(best)}
        (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    results = [json.loads(p.read_text()) for p in output.glob("exp_*/result.json")]
    results.sort(key=lambda r: r["metrics"]["metrics/mAP50-95(B)"], reverse=True)
    (output / "ranking.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Validation ranking: {output / 'ranking.json'}; test was not evaluated")


if __name__ == "__main__":
    main()
