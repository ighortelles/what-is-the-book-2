"""Apply reviewed CSV exclusions reversibly, preserving split assignments."""

import collections
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/unified"
ARCHIVE = ROOT / "data/discarded/manual_review"
CSV = ROOT / "data/annotation_review/queue_reviewed.csv"


def main() -> None:
    csv_hash = hashlib.sha256(CSV.read_bytes()).hexdigest()
    report_path = DATA / "manual_review.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["csv_sha256"] != csv_hash:
            raise RuntimeError("A different review was already applied")
        print("This CSV was already applied; no changes made")
        return
    if ARCHIVE.exists():
        raise RuntimeError("Archive exists; inspect any interrupted application before retrying")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    by_id = {str(m["id"]): m for m in manifest}
    with CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    excluded = set()
    for row in rows:
        ident = row["id"]
        if ident in seen or ident not in by_id:
            raise ValueError(f"Duplicate or unknown ID: {ident}")
        seen.add(ident)
        m = by_id[ident]
        if row["image"] != m["image"] or row["split"] != m["split"]:
            raise ValueError(f"CSV does not match manifest: {ident}")
        decision, discard = row["decision"], row["discard"]
        if decision not in ("", "manter", "excluir", "ok", "corrigir"):
            raise ValueError(f"Unknown decision: {ident}")
        if row["flags"] and not decision:
            raise ValueError(f"Flagged image still pending: {ident}")
        if (decision == "excluir") != (discard == "sim"):
            raise ValueError(f"Conflicting discard decision: {ident}")
        if discard == "sim":
            excluded.add(int(ident))
    removed = [m for m in manifest if m["id"] in excluded]
    retained = [m for m in manifest if m["id"] not in excluded]
    plan = []
    for m in removed:
        image = Path(m["image"])
        for relative in [image, image.parent.parent / "labels" / f"{image.stem}.txt"]:
            source = (DATA / relative).resolve()
            dest = (ARCHIVE / "files" / relative).resolve()
            if not source.is_relative_to(DATA.resolve()) or not dest.is_relative_to(
                ARCHIVE.resolve()
            ):
                raise ValueError("Path outside intended dataset/archive")
            if not source.is_file() or dest.exists():
                raise ValueError(f"Missing source or occupied destination: {relative}")
            plan.append((source, dest, hashlib.sha256(source.read_bytes()).hexdigest()))
    ARCHIVE.mkdir(parents=True)
    for name in ["manifest.json", "summary.json", "data.yaml"]:
        shutil.copy2(DATA / name, ARCHIVE / name)
    shutil.copy2(CSV, ARCHIVE / CSV.name)
    (ARCHIVE / "move_plan.json").write_text(
        json.dumps(
            [{"source": str(a), "destination": str(b), "sha256": h} for a, b, h in plan], indent=2
        ),
        encoding="utf-8",
    )
    for source, dest, digest in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    report = {
        "csv_sha256": csv_hash,
        "input_images": len(manifest),
        "output_images": len(retained),
        "discarded_images": len(removed),
        "discarded_ids": sorted(excluded),
        "discarded_by_split": dict(collections.Counter(m["split"] for m in removed)),
        "retained_by_split": dict(collections.Counter(m["split"] for m in retained)),
        "discarded_objects": sum(m["objects"] for m in removed),
        "retained_objects": sum(m["objects"] for m in retained),
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
    }
    summary["manual_review"] = report
    summary["output_images"] = len(retained)
    summary["groups"] = len({m["group"] for m in retained})
    summary["sources"] = {
        s: {
            split: {
                "images": sum(m["source"] == s and m["split"] == split for m in retained),
                "objects": sum(
                    m["objects"] for m in retained if m["source"] == s and m["split"] == split
                ),
            }
            for split in ["train", "val", "test"]
        }
        for s in sorted({m["source"] for m in manifest})
    }
    for name, value in [
        ("manifest.json", retained),
        ("summary.json", summary),
        ("manual_review.json", report),
    ]:
        temp = DATA / f"{name}.tmp"
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temp.replace(DATA / name)
    shutil.copy2(CSV, DATA / "provenance/manual_review.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
