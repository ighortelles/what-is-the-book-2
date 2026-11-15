"""Deterministic image-only preprocessing; boxes and split membership stay unchanged."""

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

ImageArray = NDArray[np.uint8]


def read_image(path: Path) -> ImageArray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return np.asarray(image, dtype=np.uint8)


def transform(image: ImageArray, variant: dict[str, Any]) -> ImageArray:
    """Return uint8 BGR with identical shape; normalize gradients by their own maximum."""
    op = variant["operator"]
    if op == "original":
        result = image.copy()
    elif op == "gaussian":
        k = variant["ksize"]
        result = cv2.GaussianBlur(image, (k, k), variant["sigma"])
    elif op == "median":
        result = cv2.medianBlur(image, variant["ksize"])
    elif op == "bilateral":
        result = cv2.bilateralFilter(
            image, variant["d"], variant["sigma_color"], variant["sigma_space"]
        )
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if op == "grayscale":
            edge = gray
        else:
            smooth = cv2.GaussianBlur(gray, (3, 3), 0.8)
            if op == "canny":
                edge = cv2.Canny(smooth, variant["low"], variant["high"], L2gradient=True)
            else:
                if op in {"sobel", "rgb_sobel"}:
                    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=variant["ksize"])
                    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=variant["ksize"])
                    magnitude = cv2.magnitude(dx, dy)
                elif op == "laplacian":
                    magnitude = np.abs(cv2.Laplacian(smooth, cv2.CV_32F, ksize=variant["ksize"]))
                else:
                    raise ValueError(f"Unknown operator: {op}")
                maximum = float(magnitude.max())
                edge = (
                    np.clip(magnitude * (255.0 / maximum), 0, 255).astype(np.uint8)
                    if maximum > 0
                    else np.zeros_like(gray)
                )
        result = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        if op == "rgb_sobel":
            alpha = variant["alpha"]
            result = cv2.addWeighted(image, 1 - alpha, result, alpha, 0)
    assert result.shape == image.shape and result.dtype == np.uint8
    return np.asarray(result, dtype=np.uint8)


def dataset_identity(data: Path) -> tuple[list[dict[str, Any]], str]:
    """Hash actual train/val bytes as well as their manifest. Never read test pixels."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    entries = [m for m in manifest if m["split"] in {"train", "val"}]
    assert {m["split"] for m in entries} == {"train", "val"}
    assert len({m["id"] for m in entries}) == len(entries)
    groups = {}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode())
    for m in entries:
        assert groups.setdefault(m["group"], m["split"]) == m["split"]
        image = data / m["image"]
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        for path in (image, label):
            assert path.resolve().is_relative_to(data.resolve())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return entries, digest.hexdigest()


def prepare_variant(
    data: Path,
    destination: Path,
    entries: list[dict[str, Any]],
    variant: dict[str, Any],
    signature: dict[str, Any],
) -> Path:
    """Generate lossless PNGs and unchanged labels, including after an interrupted export."""
    assert re.fullmatch(r"[a-z0-9_]+", variant["name"])
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "preprocessing.json"
    if marker.exists():
        if json.loads(marker.read_text()) != signature:
            raise RuntimeError("Prepared dataset signature differs; select another cache directory")
    else:
        if any(destination.iterdir()):
            raise RuntimeError("Refusing to write into a nonempty directory without provenance")
        marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    for m in entries:
        source = data / m["image"]
        target = destination / m["split"] / "images" / f"{source.stem}.png"
        label = source.parent.parent / "labels" / f"{source.stem}.txt"
        target_label = target.parent.parent / "labels" / label.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        output = transform(read_image(source), variant)
        success, encoded = cv2.imencode(".png", output, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        assert success
        target.write_bytes(encoded.tobytes())
        target_label.write_bytes(label.read_bytes())
    return destination


def release_variant(destination: Path, cache: Path, signature: dict[str, Any]) -> None:
    """Remove only this generated cache after successful result persistence."""
    resolved = destination.resolve()
    assert resolved.parent == cache.resolve() and not destination.is_symlink()
    assert json.loads((resolved / "preprocessing.json").read_text()) == signature
    shutil.rmtree(resolved)


def preview(data: Path, variants: list[dict[str, Any]], destination: Path) -> None:
    """Preview one training image from each source; no validation/test-based selection."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    for number, source in enumerate(sorted({m["source"] for m in manifest}), 1):
        item = next(m for m in manifest if m["source"] == source and m["split"] == "train")
        image = read_image(data / item["image"])
        sheet = Image.new("RGB", (1200, ((len(variants) + 2) // 3) * 250), "white")
        draw = ImageDraw.Draw(sheet)
        for i, variant in enumerate(variants):
            output = cv2.cvtColor(transform(image, variant), cv2.COLOR_BGR2RGB)
            tile = Image.fromarray(output)
            tile.thumbnail((390, 220))
            x, y = (i % 3) * 400, (i // 3) * 250
            sheet.paste(tile, (x, y + 25))
            draw.text((x + 5, y + 5), variant["name"], fill="black")
        sheet.save(destination / f"source_{number}_id_{item['id']}.png")
