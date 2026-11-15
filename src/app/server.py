"""Local photo detection UI, using trusted local YOLO checkpoints on CPU."""

import argparse
import base64
import importlib
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.preprocess.image_variants import transform

ASSETS = Path(__file__).parent
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def discover(storage: Path) -> dict[str, dict[str, Any]]:
    models = {}
    sources = [
        ("drive", storage / "models", False),
        ("local", storage / "runs/preprocessing_local/grid", True),
    ]
    for source, directory, nested in sources:
        for folder in sorted(directory.glob("*")):
            metadata = folder / "result.json"
            weights = folder / ("weights/best.pt" if nested else "best.pt")
            if not metadata.is_file() or not weights.is_file():
                continue
            try:
                result = json.loads(metadata.read_text(encoding="utf-8"))
                request = result.get("request", result)
                size = int(request["settings"]["imgsz"])
                variant = request.get("variant", {"name": "original", "operator": "original"})
                models[f"{source}/{folder.name}"] = {
                    "label": f"{folder.name} — {'Drive' if source == 'drive' else 'local'}",
                    "weights": weights,
                    "imgsz": size,
                    "variant": variant,
                }
            except (ValueError, KeyError, TypeError):
                continue
    return models


def decode_photo(payload: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as photo:
            if photo.width * photo.height > MAX_PIXELS:
                raise ValueError("A foto deve ter no máximo 25 megapixels.")
            rgb = np.array(ImageOps.exif_transpose(photo).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Não foi possível abrir a foto. Use JPG, PNG ou WebP.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class Detector:
    def __init__(self) -> None:
        self.key: tuple | None = None
        self.model: Any = None

    def predict(self, model: dict[str, Any], image: np.ndarray, confidence: float) -> dict:
        weights = model["weights"]
        key = (str(weights), weights.stat().st_mtime_ns)
        if self.key != key:
            self.model = None
            ultra = importlib.import_module("ultralytics")
            self.model = ultra.YOLO(str(weights))
            self.key = key
        processed = transform(image, model["variant"])
        result = self.model.predict(
            source=processed,
            imgsz=model["imgsz"],
            conf=confidence,
            iou=0.7,
            max_det=1000,
            device="cpu",
            half=False,
            verbose=False,
            save=False,
        )[0]
        boxes = []
        canvas = image.copy()
        for coordinates, score in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True
        ):
            x1, y1, x2, y2 = map(int, coordinates)
            boxes.append({"xyxy": coordinates, "confidence": score})
            thickness = max(2, round(max(image.shape[:2]) / 900))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 210, 130), thickness)
            cv2.putText(
                canvas,
                f"{score:.0%}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (70, 210, 130),
                2,
            )
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Não foi possível gerar a imagem anotada.")
        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
            "count": len(boxes),
            "boxes": boxes,
            "model": model["label"],
            "width": image.shape[1],
            "height": image.shape[0],
        }


def handler(storage: Path, detector: Detector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def json_reply(self, status: int, value: object) -> None:
            self.reply(
                status,
                json.dumps(value, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.reply(200, (ASSETS / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/models":
                models = discover(storage)
                preferred = "drive/exp_05" if "drive/exp_05" in models else next(iter(models), None)
                self.json_reply(
                    200,
                    {
                        "models": [
                            {"id": key, "label": item["label"]} for key, item in models.items()
                        ],
                        "preferred": preferred,
                    },
                )
            else:
                self.json_reply(404, {"error": "Página não encontrada."})

        def do_POST(self) -> None:
            request = urlparse(self.path)
            if request.path != "/api/detect":
                self.json_reply(404, {"error": "Página não encontrada."})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.json_reply(403, {"error": "Origem não permitida."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BYTES:
                    raise ValueError("Envie uma foto de até 20 MB.")
                query = parse_qs(request.query)
                model_id = query.get("model", [""])[0]
                confidence = float(query.get("confidence", ["0.25"])[0])
                if not 0.01 <= confidence <= 0.95:
                    raise ValueError("Confiança inválida.")
                models = discover(storage)
                if model_id not in models:
                    raise ValueError("Modelo indisponível. Atualize a lista de modelos.")
                image = decode_photo(self.rfile.read(size))
                self.json_reply(200, detector.predict(models[model_id], image, confidence))
            except ValueError as exc:
                self.json_reply(400, {"error": str(exc)})
            except Exception:
                import traceback

                traceback.print_exc()
                self.json_reply(500, {"error": "Falha na detecção. Consulte o log da aplicação."})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path("D:/what-is-the-book-2"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), handler(args.storage, Detector()))
    print(f"Aplicação: http://127.0.0.1:{args.port} — inferência em CPU", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
