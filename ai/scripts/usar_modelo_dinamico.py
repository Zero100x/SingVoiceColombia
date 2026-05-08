from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "modelo_dinamico.tflite"
LABELS_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "dynamic_labels.txt"

ROI_RATIO = 0.75
CHANNELS_PER_FRAME = 2
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 4


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
    sequence_length = max(1, channels // CHANNELS_PER_FRAME)

    if len(buffer_frames) < sequence_length:
        return None, sequence_length

    selected_frames = buffer_frames[-sequence_length:]
    stacked_channels = []
    for gray, edges in selected_frames:
        stacked_channels.append(gray)
        stacked_channels.append(edges)

    while len(stacked_channels) < channels:
        stacked_channels.append(selected_frames[-1][0])

    sample = np.stack(stacked_channels[:channels], axis=-1).astype(np.float32)

    if dtype == np.uint8:
        scale, zero_point = input_details["quantization"]
        scale = scale if scale else 1.0
        sample = sample / scale + zero_point
        sample = np.clip(sample, 0, 255).astype(np.uint8)

    return np.expand_dims(sample, axis=0).astype(dtype), sequence_length


def suavizar(predicciones):
    if not predicciones:
        return "", 0.0

    labels = [label for label, _ in predicciones]
    label = max(set(labels), key=labels.count)
    confidences = [confidence for pred, confidence in predicciones if pred == label]
    return label, float(np.mean(confidences))


def main():
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
    sequence_length = max(1, channels // CHANNELS_PER_FRAME)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se puede abrir la camara")
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
            buffer_frames.append(preparar_frame(roi, height, width))

            if len(buffer_frames) > sequence_length:
                buffer_frames.pop(0)

            sample, required_frames = preparar_input(buffer_frames, input_details)
            label = ""
            confidence = 0.0
            recognized = False

            if sample is not None:
                interpreter.set_tensor(input_details["index"], sample)
                interpreter.invoke()

                output = interpreter.get_tensor(output_details["index"])[0]
                probabilities = softmax(output)
                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index])
                label = labels[best_index] if best_index < len(labels) else "?"

                predicciones_recientes.append((label, confidence))
                if len(predicciones_recientes) > SMOOTHING_WINDOW:
                    predicciones_recientes.pop(0)

                label, confidence = suavizar(predicciones_recientes)
                recognized = confidence >= CONFIDENCE_THRESHOLD

            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (720, 145), (0, 0, 0), -1)

            if sample is None:
                text = f"Recolectando movimiento: {len(buffer_frames)}/{required_frames}"
                color = (0, 165, 255)
            else:
                text = label if recognized else "SENA DINAMICA NO RECONOCIDA"
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
