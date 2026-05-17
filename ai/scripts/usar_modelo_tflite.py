import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from entrenar_modelo_tflite import (
    HAND_LANDMARKER_TASK,
    cargar_mediapipe_tasks,
    recortar_mano_mediapipe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "modelo_alfabeto.tflite"
LABELS_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "labels.txt"

ROI_RATIO = 0.85
CONFIDENCE_THRESHOLD = 0.62
MARGIN_THRESHOLD = 0.08
HAND_MIN_AREA_RATIO = 0.004
HAND_SKIN_HINT_RATIO = 0.0015
MIN_EDGE_HINT_MEAN = 0.8
MIN_EDGE_MEAN = 1.2
STRONG_EDGE_MEAN = 3.0
MIN_LUMA_STD_DEV = 4.5
STRONG_LUMA_STD_DEV = 8.0
SMOOTHING_WINDOW = 8
MIN_STABLE_VOTES = 4
MIN_STABLE_RATIO = 0.50
DEBUG_LOG_INTERVAL = 20
NN_CONFIDENCE_THRESHOLD = 0.85
NN_MARGIN_THRESHOLD = 0.18
FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Usa el modelo TFLite de alfabeto con la camara seleccionada.",
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=None,
        help="Indice de camara a usar. Ejemplo: --camara 1",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Imprime top predicciones y detalles del modelo cada pocos frames.",
    )
    parser.add_argument(
        "--recorte-mano",
        choices=["mediapipe", "centro"],
        default="mediapipe",
        help="Usa MediaPipe para recortar la mano o el recorte central anterior.",
    )
    parser.add_argument(
        "--modelo-mano",
        default=str(HAND_LANDMARKER_TASK),
        help="Ruta al archivo hand_landmarker.task usado por MediaPipe.",
    )
    parser.add_argument(
        "--confianza-mano",
        type=float,
        default=0.5,
        help="Confianza minima de deteccion de mano. Por defecto: 0.5.",
    )
    return parser.parse_args()


def abrir_camara(indice_preferido=None):
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Automatico"),
    ]
    indices = [indice_preferido] if indice_preferido is not None else range(4)

    for indice in indices:
        for backend, nombre_backend in backends:
            cap = cv2.VideoCapture(indice, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            for _ in range(20):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Camara activa: indice {indice} ({nombre_backend})")
                    return cap
                time.sleep(0.05)

            cap.release()

    return None


def cargar_labels():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de etiquetas: {LABELS_FILE}")

    return [
        line.strip()
        for line in LABELS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def softmax(values):
    values = values.astype(np.float32)
    values = values - np.max(values)
    exps = np.exp(values)
    return exps / np.sum(exps)


def obtener_roi(frame):
    height, width = frame.shape[:2]
    size = int(min(height, width) * ROI_RATIO)
    x1 = (width - size) // 2
    y1 = (height - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def evaluar_mano_en_roi(frame):
    suavizado = cv2.GaussianBlur(frame, (7, 7), 0)
    gray = cv2.cvtColor(suavizado, cv2.COLOR_BGR2GRAY)
    faces = []
    if not FACE_CASCADE.empty():
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(48, 48),
        )

    ycrcb = cv2.cvtColor(suavizado, cv2.COLOR_BGR2YCrCb)
    mascara = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    area_total = frame.shape[0] * frame.shape[1]
    area_ratio = 0.0

    if contornos:
        area_mayor = cv2.contourArea(max(contornos, key=cv2.contourArea))
        area_ratio = area_mayor / area_total

    edge_mean = float(np.mean(crear_canal_bordes(gray)))
    luma_std = float(np.std(gray))
    has_skin_region = area_ratio >= HAND_MIN_AREA_RATIO
    has_skin_hint = area_ratio >= HAND_SKIN_HINT_RATIO
    has_edges = edge_mean >= MIN_EDGE_MEAN
    has_contrast = luma_std >= MIN_LUMA_STD_DEV
    has_strong_edges = edge_mean >= STRONG_EDGE_MEAN
    has_strong_contrast = luma_std >= STRONG_LUMA_STD_DEV
    hand_present = (
        (has_skin_region and has_edges) or
        (has_skin_region and has_contrast) or
        (has_skin_hint and has_contrast and edge_mean >= MIN_EDGE_HINT_MEAN) or
        (has_edges and has_strong_contrast) or
        (has_contrast and has_strong_edges)
    )

    stats = {
        "area_ratio": area_ratio,
        "edge_mean": edge_mean,
        "luma_std": luma_std,
        "faces": len(faces),
    }

    if len(faces) > 0:
        return False, stats

    return hand_present, stats


def crear_canal_bordes(imagen):
    """Usa el mismo canal de bordes que se genera durante entrenamiento."""
    gradiente_x = cv2.Sobel(imagen, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_32F, 0, 1, ksize=3)
    bordes = (np.abs(gradiente_x) + np.abs(gradiente_y)) * 0.5
    return np.clip(bordes, 0, 255).astype(np.float32)


def preparar_input(frame, input_details):
    shape = input_details["shape"]
    dtype = input_details["dtype"]
    height = int(shape[1])
    width = int(shape[2])
    channels = int(shape[3])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    if channels == 1:
        sample = resized[..., np.newaxis].astype(np.float32)
    else:
        canales = [resized.astype(np.float32), crear_canal_bordes(resized)]
        while len(canales) < channels:
            canales.append(resized.astype(np.float32))
        sample = np.stack(canales[:channels], axis=-1)

    if dtype == np.uint8:
        scale, zero_point = input_details["quantization"]
        scale = scale if scale else 1.0
        sample = sample / scale + zero_point
        sample = np.clip(sample, 0, 255).astype(np.uint8)

    return np.expand_dims(sample, axis=0).astype(dtype)


def suavizar(predicciones):
    if not predicciones:
        return "", 0.0, False

    labels = [label for label, _ in predicciones]
    label = max(set(labels), key=labels.count)
    votes = labels.count(label)
    stable_ratio = votes / len(labels)
    confidences = [confidence for pred, confidence in predicciones if pred == label]
    stable = votes >= MIN_STABLE_VOTES and stable_ratio >= MIN_STABLE_RATIO
    return label, float(np.mean(confidences)), stable


def mejor_prediccion(probabilities):
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else best_index
    confidence = float(probabilities[best_index])
    margin = confidence - float(probabilities[second_index])
    return best_index, confidence, margin


def elegir_indice_estable(probabilities, labels):
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    best_label = labels[best_index].strip().upper()

    if best_label != "NN":
        second_index = int(order[1]) if len(order) > 1 else best_index
        confidence = float(probabilities[best_index])
        margin = confidence - float(probabilities[second_index])
        return best_index, confidence, margin

    second_index = next(
        (int(index) for index in order if labels[int(index)].strip().upper() != "NN"),
        best_index,
    )
    nn_confidence = float(probabilities[best_index])
    second_confidence = float(probabilities[second_index])
    nn_margin = nn_confidence - second_confidence

    if nn_confidence >= NN_CONFIDENCE_THRESHOLD and nn_margin >= NN_MARGIN_THRESHOLD:
        return best_index, nn_confidence, nn_margin

    third_index = next((int(index) for index in order if int(index) != second_index), second_index)
    confidence = float(probabilities[second_index])
    margin = confidence - float(probabilities[third_index])
    return second_index, confidence, margin


def etiqueta_visible(label):
    return "\u00D1" if label.strip().upper() == "NN" else label


def top_predicciones(probabilities, labels, limit=3):
    order = np.argsort(probabilities)[::-1][:limit]
    return [
        (labels[index] if index < len(labels) else "?", float(probabilities[index]))
        for index in order
    ]


def main():
    args = parse_args()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No existe el modelo TFLite: {MODEL_FILE}")

    labels = cargar_labels()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    output_size = int(output_details["shape"][-1])

    if len(labels) != output_size:
        raise ValueError(
            "El numero de etiquetas no coincide con la salida del modelo: "
            f"{len(labels)} labels vs {output_size} salidas."
        )

    hand_detector = None
    mp_module = None
    if args.recorte_mano == "mediapipe":
        hand_detector, mp_module = cargar_mediapipe_tasks(
            Path(args.modelo_mano),
            args.confianza_mano,
        )

    cap = abrir_camara(args.camara)
    if cap is None:
        if hand_detector is not None:
            hand_detector.close()
        if args.camara is None:
            print("Error: no se pudo leer video desde ninguna camara")
        else:
            print(f"Error: no se pudo leer video desde la camara {args.camara}")
        return

    predicciones_recientes = []
    frame_index = 0

    print("Modelo TFLite cargado.")
    print(f"Clases: {labels}")
    print(f"Entrada del modelo: shape={input_details['shape']}, dtype={input_details['dtype']}")
    print(f"Salida del modelo: shape={output_details['shape']}, dtype={output_details['dtype']}")
    print(f"Recorte de mano: {args.recorte_mano}")
    print("Ubica la mano frente a la camara. Presiona ESC para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            roi_box = None
            frame_index += 1

            if args.recorte_mano == "mediapipe":
                roi = recortar_mano_mediapipe(frame, hand_detector, mp_module)
                hand_present = roi is not None
                hand_stats = {"mediapipe": hand_present}
            else:
                roi, roi_box = obtener_roi(frame)
                hand_present, hand_stats = evaluar_mano_en_roi(roi)

            if not hand_present:
                predicciones_recientes.clear()
                label = "MANO NO DETECTADA"
                confidence = 0.0
                recognized = False
            else:
                sample = preparar_input(roi, input_details)
                interpreter.set_tensor(input_details["index"], sample)
                interpreter.invoke()

                output = interpreter.get_tensor(output_details["index"])[0]
                probabilities = softmax(output)
                best_index, confidence, margin = elegir_indice_estable(probabilities, labels)
                label = labels[best_index] if best_index < len(labels) else "?"

                if args.debug and frame_index % DEBUG_LOG_INTERVAL == 0:
                    top3 = ", ".join(
                        f"{name}:{score * 100:.1f}%"
                        for name, score in top_predicciones(probabilities, labels)
                    )
                    print(
                        "Prediccion debug | "
                        f"top3=[{top3}] | margen={margin:.3f} | mano={hand_stats}"
                    )

                if confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                    predicciones_recientes.append((label, confidence))
                    if len(predicciones_recientes) > SMOOTHING_WINDOW:
                        predicciones_recientes.pop(0)

                    stable_label, stable_confidence, stable = suavizar(predicciones_recientes)
                    if stable and stable_confidence >= CONFIDENCE_THRESHOLD:
                        label = stable_label
                        confidence = stable_confidence
                        recognized = True
                    else:
                        label = "NO RECONOCIDO"
                        recognized = False
                else:
                    predicciones_recientes.clear()
                    label = "NO RECONOCIDO"
                    recognized = False

            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (620, 125), (0, 0, 0), -1)

            color = (0, 255, 0) if recognized else (0, 165, 255)
            text = etiqueta_visible(label) if recognized else label

            cv2.putText(
                frame,
                f"Sena: {text}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )
            cv2.putText(
                frame,
                f"Confianza: {confidence * 100:.1f}%",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            cv2.imshow("Reconocedor TFLite - SignVoice Colombia", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if hand_detector is not None:
            hand_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
