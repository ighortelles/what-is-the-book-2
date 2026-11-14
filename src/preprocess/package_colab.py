"""Package code, locked dependencies and the unified dataset for Colab."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    destination = ROOT / "data/colab_bundle.zip"
    files = [ROOT / name for name in ["pyproject.toml", "uv.lock", "README.md"]]
    for folder in ["src", "configs", "docs", "notebooks", "data/unified"]:
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "before_validation" not in p.parts
        )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"{destination}: {destination.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
