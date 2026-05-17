import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "modelo_dinamico.tflite"
LABELS_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "dynamic_labels.txt"

ROI_RATIO = 0.75
CHANNELS_PER_FRAME = 2
MOTION_SUMMARY_CHANNELS = 6
MOTION_SEQUENCE_LENGTH = 16
MIN_MOTION_SCORE = 2.0
CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.18
HAND_MIN_AREA_RATIO = 0.015
SMOOTHING_WINDOW = 4
UNKNOWN_LABELS = {
    "DESCONOCIDO",
    "UNKNOWN",
    "NO_ENTRENADA",
    "SENA_NO_ENTRENADA",
    "NO_SENA",
    "SIN_SENA",
    "NINGUNA",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Usa el modelo dinamico TFLite con la camara seleccionada.",
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=None,
        help="Indice de camara a usar. Ejemplo: --camara 1",
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
        raise FileNotFoundError(f"No existe el archivo de etiquetas dinamicas: {LABELS_FILE}")

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


def hay_mano_en_roi(frame):
    suavizado = cv2.GaussianBlur(frame, (7, 7), 0)
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
    if not contornos:
        return False

    area_mayor = cv2.contourArea(max(contornos, key=cv2.contourArea))
    area_total = frame.shape[0] * frame.shape[1]
    return (area_mayor / area_total) >= HAND_MIN_AREA_RATIO


def crear_canal_bordes(imagen):
    gradiente_x = cv2.Sobel(imagen, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_32F, 0, 1, ksize=3)
    bordes = (np.abs(gradiente_x) + np.abs(gradiente_y)) * 0.5
    return np.clip(bordes, 0, 255).astype(np.float32)


def preparar_frame(frame, height, width):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32), crear_canal_bordes(resized)


def preparar_input(buffer_frames, input_details):
    shape = input_details["shape"]
    dtype = input_details["dtype"]
    height = int(shape[1])
    width = int(shape[2])
    channels = int(shape[3])
    sequence_length = obtener_frames_requeridos(channels)

    if len(buffer_frames) < sequence_length:
        return None, sequence_length, None

    selected_frames = buffer_frames[-sequence_length:]
    motion_score = None

    if channels == MOTION_SUMMARY_CHANNELS:
        sample, motion_score = preparar_input_movimiento(selected_frames)
    else:
        sample = preparar_input_canales_apilados(selected_frames, channels)

    if dtype == np.uint8:
        scale, zero_point = input_details["quantization"]
        scale = scale if scale else 1.0
        sample = sample / scale + zero_point
        sample = np.clip(sample, 0, 255).astype(np.uint8)

    return np.expand_dims(sample, axis=0).astype(dtype), sequence_length, motion_score


def obtener_frames_requeridos(channels):
    if channels == MOTION_SUMMARY_CHANNELS:
        return MOTION_SEQUENCE_LENGTH

    return max(1, channels // CHANNELS_PER_FRAME)


def preparar_input_canales_apilados(selected_frames, channels):
    stacked_channels = []
    for gray, edges in selected_frames:
        stacked_channels.append(gray)
        stacked_channels.append(edges)

    while len(stacked_channels) < channels:
        stacked_channels.append(selected_frames[-1][0])

    return np.stack(stacked_channels[:channels], axis=-1).astype(np.float32)


def preparar_input_movimiento(selected_frames):
    primer_gray, primer_edges = selected_frames[0]
    ultimo_gray, ultimo_edges = selected_frames[-1]
    diferencias = [
        np.abs(selected_frames[indice + 1][0] - selected_frames[indice][0])
        for indice in range(len(selected_frames) - 1)
    ]
    movimiento = np.mean(diferencias, axis=0).astype(np.float32)
    movimiento_edges = crear_canal_bordes(movimiento)

    sample = np.stack(
        [
            primer_gray,
            primer_edges,
            ultimo_gray,
            ultimo_edges,
            movimiento,
            movimiento_edges,
        ],
        axis=-1,
    ).astype(np.float32)
    return sample, float(movimiento.mean())


def suavizar(predicciones):
    if not predicciones:
        return "", 0.0

    labels = [label for label, _ in predicciones]
    label = max(set(labels), key=labels.count)
    confidences = [confidence for pred, confidence in predicciones if pred == label]
    return label, float(np.mean(confidences))


def mejor_prediccion(probabilities):
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else best_index
    confidence = float(probabilities[best_index])
    margin = confidence - float(probabilities[second_index])
    return best_index, confidence, margin


def es_clase_desconocida(label):
    return label.strip().upper() in UNKNOWN_LABELS


def main():
    args = parse_args()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "No existe el modelo dinamico TFLite. "
            "Entrenalo con: python entrenar_modelo_dinamico.py"
        )

    labels = cargar_labels()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = input_details["shape"]
    height = int(input_shape[1])
    width = int(input_shape[2])
    channels = int(input_shape[3])
    sequence_length = obtener_frames_requeridos(channels)

    cap = abrir_camara(args.camara)
    if cap is None:
        if args.camara is None:
            print("Error: no se pudo leer video desde ninguna camara")
        else:
            print(f"Error: no se pudo leer video desde la camara {args.camara}")
        return

    buffer_frames = []
    predicciones_recientes = []

    print("Modelo dinamico TFLite cargado.")
    print(f"Clases dinamicas: {labels}")
    print(f"Frames por prediccion: {sequence_length}")
    print("Haz el movimiento completo dentro del recuadro verde. Presiona ESC para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            roi, roi_box = obtener_roi(frame)
            label = ""
            confidence = 0.0
            recognized = False
            sample = None
            required_frames = sequence_length

            if not hay_mano_en_roi(roi):
                buffer_frames.clear()
                predicciones_recientes.clear()
                label = "SIN MANO"
            else:
                buffer_frames.append(preparar_frame(roi, height, width))

                if len(buffer_frames) > sequence_length:
                    buffer_frames.pop(0)

                sample, required_frames, motion_score = preparar_input(buffer_frames, input_details)

                if sample is not None:
                    if motion_score is not None and motion_score < MIN_MOTION_SCORE:
                        predicciones_recientes.clear()
                        label = "SIN MOVIMIENTO"
                    else:
                        interpreter.set_tensor(input_details["index"], sample)
                        interpreter.invoke()

                        output = interpreter.get_tensor(output_details["index"])[0]
                        probabilities = softmax(output)
                        best_index, confidence, margin = mejor_prediccion(probabilities)
                        label = labels[best_index] if best_index < len(labels) else "?"

                        if es_clase_desconocida(label):
                            predicciones_recientes.clear()
                            label = "SENA NO ENTRENADA"
                        elif confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                            predicciones_recientes.append((label, confidence))
                            if len(predicciones_recientes) > SMOOTHING_WINDOW:
                                predicciones_recientes.pop(0)

                            label, confidence = suavizar(predicciones_recientes)
                            recognized = confidence >= CONFIDENCE_THRESHOLD
                        else:
                            label = "SIN CONFIANZA"

            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (720, 145), (0, 0, 0), -1)

            if label in {"SIN MANO", "SIN MOVIMIENTO"}:
                text = label
                color = (0, 165, 255)
            elif sample is None:
                text = f"Recolectando movimiento: {len(buffer_frames)}/{required_frames}"
                color = (0, 165, 255)
            else:
                text = label if recognized else label
                color = (0, 255, 0) if recognized else (0, 165, 255)

            cv2.putText(
                frame,
                f"Sena dinamica: {text}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confianza: {confidence * 100:.1f}%",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

            cv2.imshow("Reconocedor Dinamico - SignVoice Colombia", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
