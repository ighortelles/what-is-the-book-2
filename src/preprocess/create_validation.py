"""Split validation from training without changing the held-out test set."""

import collections
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/unified"


def main() -> None:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(m["split"] == "val" for m in manifest):
        raise RuntimeError("Validation already exists; refusing to resplit")
    train = [m for m in manifest if m["split"] == "train"]
    sources = sorted({m["source"] for m in train})
    groups = collections.defaultdict(list)
    for m in train:
        groups[m["group"]].append(m)
    grouped = list(groups.values())
    vectors = np.array([[sum(m["source"] == s for m in g) for s in sources] for g in grouped])
    target = vectors.sum(axis=0) * 0.2
    rng = random.Random(43)
    best_loss = float("inf")
    best_ids: set[int] = set()
    for _ in range(2000):
        order = list(range(len(grouped)))
        rng.shuffle(order)
        counts = np.zeros(len(sources))
        ids: set[int] = set()
        for i in order:
            current = np.sum(((counts - target) / np.maximum(target, 1)) ** 2)
            proposed = np.sum(((counts + vectors[i] - target) / np.maximum(target, 1)) ** 2)
            if proposed < current:
                counts += vectors[i]
                ids.update(m["id"] for m in grouped[i])
        loss = float(np.sum(((counts - target) / np.maximum(target, 1)) ** 2))
        if loss < best_loss:
            best_loss, best_ids = loss, ids
    if not best_ids:
        raise RuntimeError("Could not allocate validation groups")
    backup = DATA / "before_validation"
    if backup.exists():
        raise RuntimeError("Backup exists; inspect any interrupted split before retrying")
    backup.mkdir()
    for name in ["manifest.json", "summary.json", "data.yaml"]:
        (backup / name).write_bytes((DATA / name).read_bytes())
    plan = []
    for m in manifest:
        if m["id"] not in best_ids:
            continue
        old = DATA / m["image"]
        for folder, suffix in [("images", ".jpg"), ("labels", ".txt")]:
            source = (DATA / "train" / folder / old.with_suffix(suffix).name).resolve()
            dest = (DATA / "val" / folder / source.name).resolve()
            assert source.is_relative_to(DATA.resolve()) and dest.is_relative_to(DATA.resolve())
            assert source.is_file() and not dest.exists()
            plan.append((source, dest))
        m["split"] = "val"
        m["image"] = f"val/images/{old.name}"
    (backup / "move_plan.json").write_text(
        json.dumps([[str(a), str(b)] for a, b in plan], indent=2), encoding="utf-8"
    )
    for source, dest in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
    split_groups = {
        split: {m["group"] for m in manifest if m["split"] == split}
        for split in ["train", "val", "test"]
    }
    assert not split_groups["train"] & split_groups["val"]
    assert not split_groups["train"] & split_groups["test"]
    assert not split_groups["val"] & split_groups["test"]
    for m in manifest:
        assert (DATA / m["image"]).is_file()
    previous = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert [m for m in previous if m["split"] == "test"] == [
        m for m in manifest if m["split"] == "test"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    summary["validation"] = {"seed": 43, "fraction_of_previous_train": 0.2}
    summary["sources"] = {
        s: {
            split: {
                "images": sum(m["source"] == s and m["split"] == split for m in manifest),
                "objects": sum(
                    m["objects"] for m in manifest if m["source"] == s and m["split"] == split
                ),
            }
            for split in ["train", "val", "test"]
        }
        for s in sources
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: train/images\nval: val/images\n"
        "test: test/images\nnc: 1\nnames: [book]\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["sources"], indent=2))


if __name__ == "__main__":
    main()
