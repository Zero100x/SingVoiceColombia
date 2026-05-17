"""
Servidor local para pruebas de carga con JMeter.

Este modulo simula los endpoints HTTP que SignVoice Colombia podria tener en
una API futura para consultar clases, version de modelos y registrar muestras.
No reemplaza el backend productivo: es un servicio academico para ejecutar la
actividad de pruebas de carga con peticiones GET y POST reales.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DATASET_DIR = PROJECT_ROOT / "datasets" / "static_signs"
ANDROID_ASSETS_DIR = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets"

ALPHABET_CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "NN",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class LoadTestState:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.samples_received = 0


STATE = LoadTestState()


def dataset_counts() -> dict[str, int]:
    counts = {}
    for class_name in ALPHABET_CLASSES:
        class_dir = STATIC_DATASET_DIR / class_name.lower()
        if not class_dir.exists():
            class_dir = STATIC_DATASET_DIR / class_name

        if class_dir.exists():
            counts[class_name] = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            counts[class_name] = 0
    return counts


def model_version() -> dict[str, object]:
    model_file = ANDROID_ASSETS_DIR / "modelo_alfabeto.tflite"
    labels_file = ANDROID_ASSETS_DIR / "labels.txt"
    hand_file = ANDROID_ASSETS_DIR / "hand_landmarker.task"
    labels = []

    if labels_file.exists():
        labels = [
            line.strip()
            for line in labels_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return {
        "model": "modelo_alfabeto.tflite",
        "model_exists": model_file.exists(),
        "model_size_bytes": model_file.stat().st_size if model_file.exists() else 0,
        "labels_count": len(labels),
        "labels": labels,
        "hand_landmarker_exists": hand_file.exists(),
        "hand_landmarker_size_bytes": hand_file.stat().st_size if hand_file.exists() else 0,
        "version": "load-test-local-v1",
    }


def read_delay_seconds() -> float:
    raw_value = os.getenv("SIGNVOICE_API_DELAY_MS", "0").strip()
    try:
        milliseconds = max(float(raw_value), 0.0)
    except ValueError:
        milliseconds = 0.0
    return milliseconds / 1000.0


class SignVoiceLoadTestHandler(BaseHTTPRequestHandler):
    server_version = "SignVoiceLoadTestAPI/1.0"

    def do_GET(self) -> None:
        self.apply_optional_delay()
        path = urlparse(self.path).path

        if path == "/api/health":
            self.write_json(
                {
                    "status": "ok",
                    "service": "SignVoice Colombia",
                    "uptime_seconds": round(time.time() - STATE.started_at, 2),
                }
            )
            return

        if path == "/api/classes":
            counts = dataset_counts()
            self.write_json(
                {
                    "type": "alphabet",
                    "classes": ALPHABET_CLASSES,
                    "sample_counts": counts,
                    "total_samples": sum(counts.values()),
                }
            )
            return

        if path == "/api/model-version":
            self.write_json(model_version())
            return

        if path == "/api/samples/count":
            self.write_json({"samples_received": STATE.samples_received})
            return

        self.write_json({"error": "Ruta GET no encontrada", "path": path}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        self.apply_optional_delay()
        path = urlparse(self.path).path

        if path != "/api/samples":
            self.write_json({"error": "Ruta POST no encontrada", "path": path}, HTTPStatus.NOT_FOUND)
            return

        payload = self.read_json_body()
        if payload is None:
            self.write_json({"error": "JSON invalido"}, HTTPStatus.BAD_REQUEST)
            return

        class_name = str(payload.get("class_name", "")).strip().upper()
        sample_type = str(payload.get("type", "")).strip().lower()
        file_name = str(payload.get("file_name", "")).strip()

        errors = []
        if class_name not in ALPHABET_CLASSES:
            errors.append("class_name debe ser una clase del alfabeto, por ejemplo A o NN")
        if sample_type not in {"static", "dynamic"}:
            errors.append("type debe ser static o dynamic")
        if not file_name:
            errors.append("file_name es obligatorio")

        if errors:
            self.write_json({"error": "Datos invalidos", "details": errors}, HTTPStatus.BAD_REQUEST)
            return

        STATE.samples_received += 1
        self.write_json(
            {
                "id": str(uuid.uuid4()),
                "status": "accepted",
                "class_name": class_name,
                "type": sample_type,
                "file_name": file_name,
                "samples_received": STATE.samples_received,
            },
            HTTPStatus.CREATED,
        )

    def read_json_body(self) -> dict[str, object] | None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None

    def write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def apply_optional_delay(self) -> None:
        delay = read_delay_seconds()
        if delay > 0:
            time.sleep(delay)

    def log_message(self, format: str, *args: object) -> None:
        print(
            f"{self.log_date_time_string()} - {self.client_address[0]} "
            f"- {format % args}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Servidor local para pruebas de carga JMeter de SignVoice Colombia.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del servidor.")
    parser.add_argument("--port", type=int, default=8080, help="Puerto del servidor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SignVoiceLoadTestHandler)
    print(f"Servidor de pruebas SignVoice en http://{args.host}:{args.port}")
    print("Endpoints: GET /api/health, GET /api/classes, GET /api/model-version, POST /api/samples")
    print("Presiona Ctrl+C para detener.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
