import cv2
import joblib
import numpy as np
import os
from pathlib import Path
from skimage.feature import hog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ai" / "models"
MODEL_FILE = MODELS_DIR / "modelo_senna.pkl"
ENCODER_FILE = MODELS_DIR / "encoder_senna.pkl"
ROI_RATIO = 0.75
CONFIDENCE_THRESHOLD = 0.45
SMOOTHING_WINDOW = 10


if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
    print("Error: modelo no encontrado.")
    print("Verifica que existan:")
    print(f"   - {MODEL_FILE}")
    print(f"   - {ENCODER_FILE}")
    print("\nPrimero debes ejecutar: entrenar_modelo.py")
    exit()


print("Cargando modelo...")
model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
classes = encoder.classes_

if hasattr(model, "verbose"):
    model.verbose = 0

print(f"Modelo cargado con {len(classes)} senas")
print(f"Clases: {list(classes)}")


def extraer_caracteristicas(imagen):
    """Extrae caracteristicas HOG + histograma de color."""
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    hog_features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
    )

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    return np.concatenate([hog_features, hist_h, hist_s, hist_v])


def obtener_roi(frame):
    """Recorta una region cuadrada central parecida a las imagenes del dataset."""
    height, width = frame.shape[:2]
    size = int(min(height, width) * ROI_RATIO)
    x1 = (width - size) // 2
    y1 = (height - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def suavizar_prediccion(predicciones):
    if not predicciones:
        return "", 0.0

    labels = [label for label, _ in predicciones]
    label = max(set(labels), key=labels.count)
    confidence_values = [confidence for pred, confidence in predicciones if pred == label]
    return label, float(np.mean(confidence_values))


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: no se puede abrir la camara")
    exit()


predicciones_recientes = []

print("\nUbica la mano dentro del recuadro verde.")
print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi, roi_box = obtener_roi(frame)

    try:
        features = extraer_caracteristicas(roi)
        x_sample = features.reshape(1, -1)

        pred_idx = int(model.predict(x_sample)[0])
        pred_proba = model.predict_proba(x_sample)[0]

        prediccion = str(classes[pred_idx])
        confianza = float(pred_proba[pred_idx])

        predicciones_recientes.append((prediccion, confianza))
        if len(predicciones_recientes) > SMOOTHING_WINDOW:
            predicciones_recientes.pop(0)

        prediccion, confianza = suavizar_prediccion(predicciones_recientes)
    except Exception as error:
        prediccion = "Error"
        confianza = 0.0
        print(f"Error en prediccion: {error}")

    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Ubica la mano dentro del recuadro",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.rectangle(frame, (0, 0), (560, 125), (0, 0, 0), -1)

    color = (0, 255, 0) if confianza >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    etiqueta = prediccion.upper() if confianza >= CONFIDENCE_THRESHOLD else "SIN CONFIANZA"

    cv2.putText(
        frame,
        f"Senna: {etiqueta}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        color,
        3,
    )
    cv2.putText(
        frame,
        f"Confianza: {confianza * 100:.1f}%",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )

    cv2.imshow("Reconocedor de Senas LSC", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

print("\nPrograma finalizado")
