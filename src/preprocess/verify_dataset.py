"""Verify split integrity, source bytes, label counts and held-out test identity."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/unified"


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "data/audit/inventory.json").read_text(encoding="utf-8"))
    original = {r["id"]: r for r in inventory["records"]}
    previous = json.loads((DATA / "before_validation/manifest.json").read_text(encoding="utf-8"))
    review_path = DATA / "manual_review.json"
    removed: set[int] = set()
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        removed = set(review["discarded_ids"])
        archive = ROOT / review["archive"]
        before_review = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        assert manifest == [m for m in before_review if m["id"] not in removed]
        plan = json.loads((archive / "move_plan.json").read_text(encoding="utf-8"))
        for move in plan:
            assert not Path(move["source"]).exists()
            assert (
                hashlib.sha256(Path(move["destination"]).read_bytes()).hexdigest() == move["sha256"]
            )
    assert len(manifest) == len(previous) - len(removed)
    assert len({m["id"] for m in manifest}) == len(manifest)
    assert [m for m in manifest if m["split"] == "test"] == [
        m for m in previous if m["split"] == "test" and m["id"] not in removed
    ]
    group_split = {}
    count = {}
    for m in manifest:
        group = m["group"]
        assert group_split.setdefault(group, m["split"]) == m["split"]
        image = DATA / m["image"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == original[m["id"]]["file_hash"]
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        lines = label.read_text().splitlines()
        assert len(lines) == m["objects"]
        for line in lines:
            cls, x, y, w, h = map(float, line.split())
            assert cls == 0 and 0 < w <= 1 and 0 < h <= 1
            assert min(x - w / 2, y - h / 2) >= -1e-6
            assert max(x + w / 2, y + h / 2) <= 1 + 1e-6
        count[m["split"]] = count.get(m["split"], 0) + 1
    for split, expected in count.items():
        assert len(list((DATA / split / "images").glob("*.jpg"))) == expected
        assert len(list((DATA / split / "labels").glob("*.txt"))) == expected
    print(
        json.dumps(
            {
                "verified_images": len(manifest),
                "splits": count,
                "test_changes_match_review": True,
                "manual_discards": len(removed),
                "shared_groups": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
