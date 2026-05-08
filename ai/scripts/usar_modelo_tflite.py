import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "modelo_alfabeto.tflite"
LABELS_FILE = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets" / "labels.txt"

ROI_RATIO = 0.75
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_WINDOW = 8


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
        return "", 0.0

    labels = [label for label, _ in predicciones]
    label = max(set(labels), key=labels.count)
    confidences = [confidence for pred, confidence in predicciones if pred == label]
    return label, float(np.mean(confidences))


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No existe el modelo TFLite: {MODEL_FILE}")

    labels = cargar_labels()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se puede abrir la camara")
        return

    predicciones_recientes = []

    print("Modelo TFLite cargado.")
    print(f"Clases: {labels}")
    print("Ubica la mano dentro del recuadro verde. Presiona ESC para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            roi, roi_box = obtener_roi(frame)

            sample = preparar_input(roi, input_details)
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
            cv2.rectangle(frame, (0, 0), (620, 125), (0, 0, 0), -1)

            color = (0, 255, 0) if recognized else (0, 165, 255)
            text = label if recognized else "SENA NO RECONOCIDA"

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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
