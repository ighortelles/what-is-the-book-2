"""Audit and merge local YOLO datasets. Requires numpy, Pillow, opencv-python."""

import argparse
import collections
import hashlib
import io
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
AUDIT = ROOT / "data/audit"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_raw(path: Path) -> bytes:
    """Read raw bytes, falling back to the source ZIP for inaccessible paths."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        rel = path.relative_to(RAW)
        with zipfile.ZipFile(ROOT / "data/zips" / f"{rel.parts[0]}.zip") as archive:
            return archive.read(Path(*rel.parts[1:]).as_posix())


def audit() -> None:
    """Validate annotations and inventory exact and perceptual similarities."""
    records, errors, thumbnails = [], [], []
    for path in sorted(RAW.rglob("*.jpg")):
        rel = path.relative_to(RAW)
        label = path.parent.parent / "labels" / (path.stem + ".txt")
        try:
            boxes, polygons = [], 0
            recovered = False
            if label.exists():
                label_text = label.read_text()
            else:
                with zipfile.ZipFile(ROOT / "data/zips" / f"{rel.parts[0]}.zip") as archive:
                    label_text = archive.read(
                        label.relative_to(RAW / rel.parts[0]).as_posix()
                    ).decode()
                recovered = True
            for line in label_text.splitlines():
                if not line.strip():
                    continue
                v = list(map(float, line.split()))
                assert v[0] == 0 and np.isfinite(v).all(), "invalid class or nonfinite coordinates"
                assert all(-1e-6 <= x <= 1 + 1e-6 for x in v[1:]), "coordinates outside image"
                if len(v) == 5:
                    x, y, w, h = v[1:]
                    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                else:
                    assert len(v) >= 7 and len(v) % 2 == 1, "invalid polygon"
                    xy = np.array(v[1:]).reshape(-1, 2)
                    x1, y1 = xy.min(axis=0)
                    x2, y2 = xy.max(axis=0)
                    polygons += 1
                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)
                assert x2 > x1 and y2 > y1, "zero-area box"
                box = [float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1)]
                if box not in boxes:
                    boxes.append(box)
            image_bytes = read_raw(path)
            with Image.open(io.BytesIO(image_bytes)) as im:
                rgb = im.convert("RGB")
                gray = np.array(
                    rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
                )
                dct = cv2.dct(gray)[:8, :8].flatten()[1:]
                ph = sum(int(b) << i for i, b in enumerate(dct > np.median(dct)))
                records.append(
                    dict(
                        id=len(records),
                        path=rel.as_posix(),
                        source=rel.parts[0],
                        label_from_zip=recovered,
                        original_split=rel.parts[1],
                        family=path.stem.split(".rf.")[0],
                        width=rgb.width,
                        height=rgb.height,
                        file_hash=sha(image_bytes),
                        pixel_hash=sha(str(rgb.size).encode() + rgb.tobytes()),
                        phash=ph,
                        boxes=boxes,
                        polygons=polygons,
                    )
                )
                thumbnails.append(
                    np.array(rgb.resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32)
                )
        except Exception as exc:
            errors.append(dict(path=rel.as_posix(), error=str(exc)))
    pairs = []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            distance = (a["phash"] ^ b["phash"]).bit_count()
            if distance <= 6:
                pairs.append([a["id"], b["id"], distance, a["pixel_hash"] == b["pixel_hash"]])
    near_duplicates = []
    for a, b, d, exact in pairs:
        mae = float(np.mean(np.abs(thumbnails[a] - thumbnails[b])))
        aspect_a = records[a]["width"] / records[a]["height"]
        aspect_b = records[b]["width"] / records[b]["height"]
        if not exact and d <= 2 and mae <= 3.0 and abs(aspect_a / aspect_b - 1) <= 0.01:
            near_duplicates.append(
                dict(a=a, b=b, phash_distance=d, mean_absolute_rgb_difference=mae)
            )
    result = dict(
        records=records, errors=errors, perceptual_pairs=pairs, near_duplicates=near_duplicates
    )
    AUDIT.mkdir(exist_ok=True)
    (AUDIT / "inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                images=len(records),
                errors=errors[:20],
                sources=dict(collections.Counter(r["source"] for r in records)),
                polygons=sum(r["polygons"] for r in records),
                exact_duplicate_excess=len(records) - len(set(r["pixel_hash"] for r in records)),
                similar_pairs=len(pairs),
                near_duplicate_pairs=len(near_duplicates),
            ),
            indent=2,
        )
    )
    # Most informative cross-name/cross-source pairs, for human visual review.
    candidates = [
        p
        for p in pairs
        if not p[3]
        and (
            records[p[0]]["family"] != records[p[1]]["family"]
            or records[p[0]]["source"] != records[p[1]]["source"]
        )
    ]
    candidates.sort(key=lambda p: p[2])
    (AUDIT / "review_pairs.json").write_text(json.dumps(candidates), encoding="utf-8")
    for page in range((min(len(candidates), 120) + 11) // 12):
        sheet = Image.new("RGB", (1000, 6 * 170), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (a, b, d, _) in enumerate(candidates[page * 12 : page * 12 + 12]):
            x = (k % 2) * 500
            y = (k // 2) * 170
            for j, idx in enumerate([a, b]):
                with Image.open(io.BytesIO(read_raw(RAW / records[idx]["path"]))) as im:
                    im.thumbnail((240, 140))
                    sheet.paste(im, (x + j * 250, y + 25))
            draw.text((x, y), f"{a} / {b}   pHash distance={d}", fill="black")
        sheet.save(AUDIT / f"review_{page:02}.jpg")


def build() -> None:
    """Deduplicate, split groups by origin and export detection annotations."""
    inventory = json.loads((AUDIT / "inventory.json").read_text())
    assert not inventory["errors"], "Resolve audit errors first"
    records = inventory["records"]
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(b)] = find(a)

    families = {}
    pixels = {}
    removed = []
    # First group exact duplicates; near-identical copies are handled below.
    for r in sorted(records, key=lambda r: (-len(r["boxes"]), r["path"])):
        i = r["id"]
        key = (r["source"], r["family"])
        if key in families:
            union(i, families[key])
        else:
            families[key] = i
        h = r["pixel_hash"]
        if h in pixels:
            keeper = pixels[h]
            union(i, keeper)
            removed.append(
                dict(
                    removed=i,
                    kept=keeper,
                    annotations_differ=r["boxes"] != records[keeper]["boxes"],
                )
            )
        else:
            pixels[h] = i
    duplicate_parent = list(range(len(records)))

    def duplicate_find(i: int) -> int:
        while duplicate_parent[i] != i:
            duplicate_parent[i] = duplicate_parent[duplicate_parent[i]]
            i = duplicate_parent[i]
        return i

    for pair in inventory["near_duplicates"]:
        a, b = pair["a"], pair["b"]
        duplicate_parent[duplicate_find(b)] = duplicate_find(a)
    for item in removed:
        duplicate_parent[duplicate_find(item["removed"])] = duplicate_find(item["kept"])
    clusters = collections.defaultdict(list)
    for r in records:
        clusters[duplicate_find(r["id"])].append(r["id"])
    removed = []
    keep = set()
    for cluster in clusters.values():
        keeper = min(
            cluster,
            key=lambda i: (
                -len(records[i]["boxes"]),
                -records[i]["width"] * records[i]["height"],
                records[i]["path"],
            ),
        )
        keep.add(keeper)
        for i in cluster:
            if i != keeper:
                union(i, keeper)
                removed.append(
                    dict(
                        removed=i,
                        kept=keeper,
                        kind="exact"
                        if records[i]["pixel_hash"] == records[keeper]["pixel_hash"]
                        else "near_identical",
                        annotations_differ=records[i]["boxes"] != records[keeper]["boxes"],
                    )
                )
    for a, b, _, _ in inventory["perceptual_pairs"]:
        union(a, b)
    groups = collections.defaultdict(list)
    for i in sorted(keep):
        groups[find(i)].append(i)
    sources = sorted(set(r["source"] for r in records))
    totals = np.array([sum(records[i]["source"] == s for i in keep) for s in sources])
    target = totals * 0.2

    def score(counts: NDArray[np.float64]) -> float:
        return float(np.sum(((counts - target) / np.maximum(target, 1)) ** 2))

    # Search deterministic group assignments for closest per-source 80/20 split.
    rng = random.Random(42)
    best = None
    group_list = list(groups.values())
    for _ in range(2000):
        rng.shuffle(group_list)
        counts = np.zeros(len(sources))
        chosen = []
        for group in group_list:
            delta = np.array([sum(records[i]["source"] == s for i in group) for s in sources])
            if score(counts + delta) < score(counts):
                counts += delta
                chosen.append(group)
        loss = score(counts)
        if best is None or loss < best[0]:
            best = (loss, {i for g in chosen for i in g})
    assert best is not None, "No split candidate was generated"
    test = best[1]
    out = ROOT / "data/unified"
    assert not out.exists(), "Output exists; refusing to overwrite"
    for split in ["train", "test"]:
        for folder in ["images", "labels"]:
            (out / split / folder).mkdir(parents=True)
    manifest = []
    for i in sorted(keep):
        r = records[i]
        split = "test" if i in test else "train"
        name = f"s{sources.index(r['source']) + 1}_{i:05d}"
        (out / split / "images" / f"{name}.jpg").write_bytes(read_raw(RAW / r["path"]))
        (out / split / "labels" / f"{name}.txt").write_text(
            "".join("0 " + " ".join(f"{v:.9f}" for v in b) + "\n" for b in r["boxes"])
        )
        manifest.append(
            dict(
                id=i,
                source=r["source"],
                original=r["path"],
                split=split,
                group=find(i),
                image=f"{split}/images/{name}.jpg",
                objects=len(r["boxes"]),
                pixel_hash=r["pixel_hash"],
            )
        )
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\ntest: test/images\nnc: 1\nnames: [book]\n",
        encoding="utf-8",
    )
    stats = {
        s: {
            sp: dict(
                images=sum(m["source"] == s and m["split"] == sp for m in manifest),
                objects=sum(
                    m["objects"] for m in manifest if m["source"] == s and m["split"] == sp
                ),
            )
            for sp in ["train", "test"]
        }
        for s in sources
    }
    summary = dict(
        seed=42,
        test_fraction_target=0.2,
        input_images=len(records),
        output_images=len(keep),
        removed_duplicates=len(removed),
        removed_exact_duplicates=sum(r["kind"] == "exact" for r in removed),
        annotation_differences=sum(r["annotations_differ"] for r in removed),
        labels_recovered_from_zip=sum(r["label_from_zip"] for r in records),
        groups=len(groups),
        sources=stats,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "duplicates.json").write_text(json.dumps(removed, indent=2), encoding="utf-8")
    for source in sources:
        dest = out / "provenance" / source
        dest.mkdir(parents=True)
        for p in (RAW / source).glob("*"):
            if p.is_file():
                shutil.copy2(p, dest / p.name)
    assert not (
        {m["group"] for m in manifest if m["split"] == "train"}
        & {m["group"] for m in manifest if m["split"] == "test"}
    )
    assert len({m["pixel_hash"] for m in manifest}) == len(manifest)
    for m in manifest:
        p = out / m["image"]
        assert p.exists()
        lines = (p.parent.parent / "labels" / f"{p.stem}.txt").read_text().splitlines()
        assert len(lines) == m["objects"] and all(len(line.split()) == 5 for line in lines)
    sheet = Image.new("RGB", (1200, 4 * 240), "white")
    draw = ImageDraw.Draw(sheet)
    for row, source in enumerate(sources):
        for col, split in enumerate(["train", "test"]):
            examples = [
                m
                for m in manifest
                if m["source"] == source and m["split"] == split and m["objects"]
            ]
            for j, m in enumerate(random.Random(42).sample(examples, min(2, len(examples)))):
                r = records[m["id"]]
                with Image.open(out / m["image"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((295, 210))
                    overlay = ImageDraw.Draw(im)
                    for x, y, w, h in r["boxes"]:
                        overlay.rectangle(
                            (
                                (x - w / 2) * im.width,
                                (y - h / 2) * im.height,
                                (x + w / 2) * im.width,
                                (y + h / 2) * im.height,
                            ),
                            outline="lime",
                            width=1,
                        )
                    x0 = (col * 2 + j) * 300
                    y0 = row * 240
                    sheet.paste(im, (x0, y0 + 25))
                    draw.text((x0, y0), f"S{row + 1} {split} id={m['id']}", fill="black")
    sheet.save(AUDIT / "unified_preview.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "build"])
    args = parser.parse_args()
    audit() if args.stage == "audit" else build()
