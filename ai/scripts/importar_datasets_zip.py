import argparse
import json
import re
import tempfile
import unicodedata
import zipfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
STATIC_SIGNS_DIR = DATASETS_DIR / "static_signs"
DYNAMIC_SIGNS_DIR = DATASETS_DIR / "dynamic_signs"

DEFAULT_LSC70_SUBSETS = {"LSC70ANH", "LSC70W"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".m4v", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MOTION_IMAGE_SIZE = (96, 96)
MOTION_ROI_RATIO = 0.75
DEFAULT_MOTION_THRESHOLD = 2.0

LABEL_ALIASES = {
    "dies": "diez",
}


def nombre_seguro(nombre):
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = "".join(char for char in nombre if not unicodedata.combining(char))
    nombre = nombre.strip().lower()
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = re.sub(r"[^a-z0-9_-]", "", nombre)
    nombre = re.sub(r"_+", "_", nombre).strip("_")
    return LABEL_ALIASES.get(nombre, nombre)


def archivo_seguro(nombre):
    stem = nombre_seguro(Path(nombre).stem)
    suffix = Path(nombre).suffix.lower()
    return f"{stem}{suffix}"


def limpiar_frames(carpeta):
    for frame_path in carpeta.glob("frame_*.jpg"):
        frame_path.unlink()


def escribir_jpg(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"No se pudo codificar frame: {path}")
    path.write_bytes(encoded.tobytes())


def leer_frames_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames


def seleccionar_frames(frames, cantidad):
    if not frames:
        return []

    indices = np.linspace(0, len(frames) - 1, cantidad)
    indices = np.rint(indices).astype(int)
    return [frames[indice] for indice in indices]


def preparar_frame_movimiento(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    roi_size = int(min(height, width) * MOTION_ROI_RATIO)
    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2
    gray = gray[y1 : y1 + roi_size, x1 : x1 + roi_size]
    return cv2.resize(gray, MOTION_IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def calcular_scores_movimiento(frames):
    if len(frames) < 2:
        return []

    procesados = [preparar_frame_movimiento(frame) for frame in frames]
    return [
        float(np.abs(procesados[index + 1] - procesados[index]).mean())
        for index in range(len(procesados) - 1)
    ]


def expandir_segmento(start, end, total_frames, min_frames):
    while (end - start + 1) < min_frames and (start > 0 or end < total_frames - 1):
        if start > 0:
            start -= 1
        if (end - start + 1) >= min_frames:
            break
        if end < total_frames - 1:
            end += 1

    return start, end


def seleccionar_segmento_activo(frames, cantidad, motion_threshold=DEFAULT_MOTION_THRESHOLD):
    if len(frames) <= cantidad:
        return seleccionar_frames(frames, cantidad), {
            "motion_trimmed": False,
            "reason": "video_short",
            "start": 0,
            "end": max(0, len(frames) - 1),
        }

    scores = calcular_scores_movimiento(frames)
    if not scores:
        return seleccionar_frames(frames, cantidad), {
            "motion_trimmed": False,
            "reason": "no_motion_scores",
            "start": 0,
            "end": max(0, len(frames) - 1),
        }

    threshold = max(motion_threshold, float(np.percentile(scores, 65)))
    active = [index for index, score in enumerate(scores) if score >= threshold]

    if not active:
        return seleccionar_frames(frames, cantidad), {
            "motion_trimmed": False,
            "reason": "no_active_segment",
            "threshold": threshold,
            "start": 0,
            "end": len(frames) - 1,
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
        }

    padding = max(2, cantidad // 8)
    start = max(0, active[0] - padding)
    end = min(len(frames) - 1, active[-1] + 1 + padding)
    start, end = expandir_segmento(start, end, len(frames), cantidad)
    segment = frames[start : end + 1]

    metadata = {
        "motion_trimmed": True,
        "threshold": threshold,
        "start": int(start),
        "end": int(end),
        "original_frames": len(frames),
        "segment_frames": len(segment),
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
    }
    return seleccionar_frames(segment, cantidad), metadata


def siguiente_muestra_path(clase_dir):
    clase_dir.mkdir(parents=True, exist_ok=True)
    numero = 1
    while True:
        muestra_path = clase_dir / f"muestra_{numero:03d}"
        if not muestra_path.exists():
            return muestra_path
        numero += 1


def buscar_muestra_por_origen(clase_dir, source_id):
    if not clase_dir.exists():
        return None

    for muestra_path in sorted(path for path in clase_dir.iterdir() if path.is_dir()):
        source_file = muestra_path / ".source.txt"
        if source_file.exists() and source_file.read_text(encoding="utf-8").strip() == source_id:
            return muestra_path

    return None


def importar_lsc70(zip_path, subsets, dry_run=False):
    if not zip_path.exists():
        return {"zip": zip_path.name, "imported": 0, "skipped": 0, "missing": True}

    imported = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as zf:
        image_infos = [
            info
            for info in zf.infolist()
            if not info.is_dir()
            and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]

        for index, info in enumerate(image_infos, start=1):
            parts = [part for part in info.filename.split("/") if part]
            if len(parts) < 5:
                skipped += 1
                continue

            _, subset, person, clase, filename = parts[:5]
            if subset not in subsets:
                skipped += 1
                continue

            clase_segura = nombre_seguro(clase)
            if not clase_segura:
                skipped += 1
                continue

            output_dir = STATIC_SIGNS_DIR / clase_segura
            output_name = archivo_seguro(f"{subset}_{person}_{filename}")
            output_path = output_dir / output_name

            if output_path.exists():
                skipped += 1
                continue

            imported += 1
            if not dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(zf.read(info))

            if imported % 1000 == 0:
                print(f"LSC70: {imported} imagenes importadas...")

    return {"zip": zip_path.name, "imported": imported, "skipped": skipped, "missing": False}


def iter_video_entries(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            if Path(info.filename).suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            parts = [part for part in info.filename.split("/") if part]
            if len(parts) < 5:
                continue

            root, subject, category, clase, filename = parts[:5]
            yield zf, info, root, subject, category, clase, filename


def importar_videos(
    zip_path,
    min_frames=30,
    dry_run=False,
    overwrite=False,
    trim_motion=True,
    motion_threshold=DEFAULT_MOTION_THRESHOLD,
):
    if not zip_path.exists():
        return {"zip": zip_path.name, "imported": 0, "skipped": 0, "missing": True}

    imported = 0
    skipped = 0
    failed = 0

    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        video_infos = [
            info
            for info in zf.infolist()
            if not info.is_dir()
            and Path(info.filename).suffix.lower() in VIDEO_EXTENSIONS
        ]

        for info in video_infos:
            parts = [part for part in info.filename.split("/") if part]
            if len(parts) < 5:
                skipped += 1
                continue

            root, subject, category, clase, filename = parts[:5]
            clase_segura = nombre_seguro(clase)
            if not clase_segura:
                skipped += 1
                continue

            source_id = info.filename
            clase_dir = DYNAMIC_SIGNS_DIR / clase_segura
            muestra_path = buscar_muestra_por_origen(clase_dir, source_id)
            if muestra_path is not None:
                current_frames = len(list(muestra_path.glob("frame_*.jpg")))
                if current_frames >= min_frames and not overwrite:
                    skipped += 1
                    continue
            else:
                muestra_path = siguiente_muestra_path(clase_dir)

            imported += 1
            if dry_run:
                continue

            tmp_video = tmp_dir / archivo_seguro(f"{root}_{subject}_{category}_{clase}_{filename}")
            tmp_video.write_bytes(zf.read(info))

            try:
                frames = leer_frames_video(tmp_video)
                if trim_motion:
                    seleccionados, motion_metadata = seleccionar_segmento_activo(
                        frames,
                        min_frames,
                        motion_threshold=motion_threshold,
                    )
                else:
                    seleccionados = seleccionar_frames(frames, min_frames)
                    motion_metadata = {
                        "motion_trimmed": False,
                        "reason": "disabled",
                        "original_frames": len(frames),
                    }

                if len(seleccionados) != min_frames:
                    raise RuntimeError(
                        f"Video sin frames suficientes para muestrear: {info.filename}"
                    )

                muestra_path.mkdir(parents=True, exist_ok=True)
                limpiar_frames(muestra_path)
                for frame_index, frame in enumerate(seleccionados, start=1):
                    escribir_jpg(muestra_path / f"frame_{frame_index:03d}.jpg", frame)

                (muestra_path / ".source.txt").write_text(source_id, encoding="utf-8")
                motion_metadata = {"source": source_id, **motion_metadata}
                (muestra_path / ".motion.json").write_text(
                    json.dumps(motion_metadata, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception as error:
                failed += 1
                print(f"ERROR importando {info.filename}: {error}")
            finally:
                tmp_video.unlink(missing_ok=True)

            if imported % 100 == 0:
                print(f"{zip_path.name}: {imported} videos procesados...")

    return {
        "zip": zip_path.name,
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "missing": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapta ZIPs externos al formato local de datasets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="No escribe archivos.")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="Frames por muestra dinamica importada.",
    )
    parser.add_argument(
        "--skip-lsc70",
        action="store_true",
        help="No importa imagenes estaticas desde LSC70.zip.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="No importa videos dinamicos desde los ZIP de videos.",
    )
    parser.add_argument(
        "--overwrite-videos",
        action="store_true",
        help="Reescribe muestras dinamicas ya importadas desde los ZIP.",
    )
    parser.add_argument(
        "--no-trim-motion",
        action="store_true",
        help="No recorta los videos al segmento con movimiento.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=DEFAULT_MOTION_THRESHOLD,
        help="Movimiento minimo para considerar activo un segmento de video.",
    )
    parser.add_argument(
        "--lsc70-subsets",
        nargs="+",
        default=sorted(DEFAULT_LSC70_SUBSETS),
        help="Subconjuntos de LSC70 a importar.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    STATIC_SIGNS_DIR.mkdir(parents=True, exist_ok=True)
    DYNAMIC_SIGNS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    if not args.skip_lsc70:
        summaries.append(
            importar_lsc70(
                DATASETS_DIR / "LSC70.zip",
                set(args.lsc70_subsets),
                dry_run=args.dry_run,
            )
        )

    if not args.skip_videos:
        for zip_name in ["Videos Colors.zip", "Videos Numbers.zip"]:
            summaries.append(
                importar_videos(
                    DATASETS_DIR / zip_name,
                    min_frames=args.min_frames,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite_videos,
                    trim_motion=not args.no_trim_motion,
                    motion_threshold=args.motion_threshold,
                )
            )

    print("\nResumen de importacion")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
