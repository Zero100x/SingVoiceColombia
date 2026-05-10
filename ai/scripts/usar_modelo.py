import argparse
import cv2
import joblib
import numpy as np
import os
import time
from pathlib import Path
from skimage.feature import hog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ai" / "models"
MODEL_FILE = MODELS_DIR / "modelo_senna.pkl"
ENCODER_FILE = MODELS_DIR / "encoder_senna.pkl"
ROI_RATIO = 0.75
CONFIDENCE_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.18
HAND_MIN_AREA_RATIO = 0.015
SMOOTHING_WINDOW = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Usa el modelo clasico con la camara seleccionada.",
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=None,
        help="Indice de camara a usar. Ejemplo: --camara 1",
    )
    return parser.parse_args()


args = parse_args()


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


def suavizar_prediccion(predicciones):
    if not predicciones:
        return "", 0.0

    labels = [label for label, _ in predicciones]
    label = max(set(labels), key=labels.count)
    confidence_values = [confidence for pred, confidence in predicciones if pred == label]
    return label, float(np.mean(confidence_values))


def mejor_prediccion(probabilidades):
    orden = np.argsort(probabilidades)[::-1]
    mejor = int(orden[0])
    segunda = int(orden[1]) if len(orden) > 1 else mejor
    confianza = float(probabilidades[mejor])
    margen = confianza - float(probabilidades[segunda])
    return mejor, confianza, margen


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


cap = abrir_camara(args.camara)

if cap is None:
    if args.camara is None:
        print("Error: no se pudo leer video desde ninguna camara")
    else:
        print(f"Error: no se pudo leer video desde la camara {args.camara}")
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
        if not hay_mano_en_roi(roi):
            predicciones_recientes.clear()
            prediccion = "SIN MANO"
            confianza = 0.0
        else:
            features = extraer_caracteristicas(roi)
            x_sample = features.reshape(1, -1)

            pred_proba = model.predict_proba(x_sample)[0]
            pred_idx, confianza, margen = mejor_prediccion(pred_proba)
            prediccion = str(classes[pred_idx])

            if confianza >= CONFIDENCE_THRESHOLD and margen >= MARGIN_THRESHOLD:
                predicciones_recientes.append((prediccion, confianza))
                if len(predicciones_recientes) > SMOOTHING_WINDOW:
                    predicciones_recientes.pop(0)

                prediccion, confianza = suavizar_prediccion(predicciones_recientes)
            else:
                prediccion = "SIN CONFIANZA"
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

    reconocido = (
        prediccion not in {"SIN MANO", "SIN CONFIANZA", "Error"}
        and confianza >= CONFIDENCE_THRESHOLD
    )
    color = (0, 255, 0) if reconocido else (0, 165, 255)
    etiqueta = prediccion.upper() if reconocido else prediccion

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
