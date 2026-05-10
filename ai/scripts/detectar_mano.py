import argparse
import os
import time

import cv2
import numpy as np


os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def cargar_mediapipe_legacy():
    """Carga la API antigua de MediaPipe si esta disponible."""
    try:
        import mediapipe as mp
    except ImportError:
        return None, None

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        return None, None

    return solutions.hands, solutions.drawing_utils


def detectar_mano_opencv(frame):
    """Detector basico por color/contorno para validar camara sin mp.solutions."""
    suavizado = cv2.GaussianBlur(frame, (7, 7), 0)
    ycrcb = cv2.cvtColor(suavizado, cv2.COLOR_BGR2YCrCb)

    mascara = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.dilate(mascara, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contornos:
        return False

    contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno)
    if area < 3000:
        return False

    x, y, ancho, alto = cv2.boundingRect(contorno)
    hull = cv2.convexHull(contorno)

    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
    cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
    cv2.putText(
        frame,
        "Mano detectada",
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta una mano usando la camara seleccionada.",
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=None,
        help="Indice de camara a usar. Ejemplo: --camara 1",
    )
    return parser.parse_args()


def abrir_camara(indice_preferido=None):
    """Prueba varias combinaciones comunes de camara en Windows."""
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


def main():
    args = parse_args()
    mp_hands, mp_draw = cargar_mediapipe_legacy()
    usar_mediapipe = mp_hands is not None and mp_draw is not None

    cap = abrir_camara(args.camara)
    if cap is None:
        if args.camara is None:
            print("Error: No se pudo leer video desde ninguna camara.")
        else:
            print(f"Error: No se pudo leer video desde la camara {args.camara}.")
        print("Revisa que la camara no este abierta en otra aplicacion.")
        print("Tambien puedes probar otro puerto USB o dar permisos de camara a Python.")
        return

    hands = None
    if usar_mediapipe:
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        print("Detector iniciado con MediaPipe solutions.")
    else:
        print(
            "MediaPipe solutions no esta disponible en esta version. "
            "Usando detector basico con OpenCV."
        )

    try:
        fallos_lectura = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                fallos_lectura += 1
                if fallos_lectura >= 30:
                    print("La camara se abrio, pero dejo de entregar frames.")
                    break
                time.sleep(0.03)
                continue

            fallos_lectura = 0

            frame = cv2.flip(frame, 1)

            if usar_mediapipe:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )
            else:
                detectar_mano_opencv(frame)

            cv2.putText(
                frame,
                "ESC: salir",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Detector de mano", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if hands is not None:
            hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
