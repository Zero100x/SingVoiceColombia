import argparse
import cv2
import os
import re
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
dataset_path = PROJECT_ROOT / "datasets" / "static_signs"
dataset_dinamico_path = PROJECT_ROOT / "datasets" / "dynamic_signs"
camera_index = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Captura imagenes o secuencias para el dataset.",
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=None,
        help="Indice de camara a usar. Ejemplo: --camara 1",
    )
    return parser.parse_args()


def nombre_seguro(nombre):
    """Convierte el nombre de la sena en una carpeta segura y sin espacios."""
    nombre = nombre.strip().lower()
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = re.sub(r"[^a-z0-9_-]", "", nombre)
    return nombre


def crear_carpeta_si_no_existe(path):
    if not os.path.exists(path):
        os.makedirs(path)


def abrir_camara():
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Automatico"),
    ]
    indices = [camera_index] if camera_index is not None else range(4)

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

    if camera_index is None:
        print("Error: No se pudo leer video desde ninguna camara")
    else:
        print(f"Error: No se pudo leer video desde la camara {camera_index}")
    return None


def capturar_sena_estatica():
    # Crear carpeta si no existe
    crear_carpeta_si_no_existe(dataset_path)

    # Pedir el nombre de la sena
    senna = input("Que sena quieres capturar? (ejemplo: hola, gracias, si, no): ")
    senna = nombre_seguro(senna)

    if not senna:
        print("Debes escribir el nombre de la sena")
        return

    # Crear carpeta para la sena
    senna_path = os.path.join(dataset_path, senna)
    if not os.path.exists(senna_path):
        os.makedirs(senna_path)
        print(f"Carpeta creada: {senna_path}")
    else:
        print(f"Carpeta ya existe: {senna_path}")

    # Contar imagenes existentes
    img_count = len([f for f in os.listdir(senna_path) if f.endswith(".jpg")])
    print(f"Imagenes existentes: {img_count}")

    # Abrir camara
    cap = abrir_camara()
    if cap is None:
        return

    # Contador de imagenes capturadas
    captured = 0
    total_images = 30  # Capturar 30 imagenes por sena

    print("\nPresiona 'ESPACIO' para capturar imagen")
    print("Presiona 'ESC' para salir")
    print(f"Debes capturar {total_images} imagenes")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espejo horizontal
        frame = cv2.flip(frame, 1)
        frame_limpio = frame.copy()
        h, w, c = frame.shape

        # Hacer la imagen mas clara (mejorar iluminacion)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_enhanced = clahe.apply(frame_gray)

        # Dibujar circulo en el centro (referencia para la mano)
        cv2.circle(frame, (w // 2, h // 2), 50, (0, 255, 0), 2)

        # Texto en pantalla
        cv2.putText(
            frame,
            f"Senna: {senna.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Capturadas: {captured}/{total_images}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        if captured >= total_images:
            cv2.putText(
                frame,
                "LISTO! Presiona ESC para terminar",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Mostrar
        cv2.imshow("Capturador de Dataset", frame)

        # Capturar con ESPACIO
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # ESPACIO
            img_count += 1
            while os.path.exists(os.path.join(senna_path, f"{img_count}.jpg")):
                img_count += 1

            captured += 1
            filename = os.path.join(senna_path, f"{img_count}.jpg")
            cv2.imwrite(filename, frame_limpio)
            print(f"Imagen guardada: {img_count}.jpg")
            time.sleep(0.3)  # Pequena pausa

        # Salir con ESC
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nCaptura completada!")
    print(f"Total de imagenes guardadas: {captured}")
    print(f"Ubicacion: {senna_path}")


def siguiente_muestra_path(senna_path):
    """Devuelve una carpeta muestra_### nueva, sin sobrescribir datos existentes."""
    crear_carpeta_si_no_existe(senna_path)
    numero = 1

    while True:
        muestra_path = os.path.join(senna_path, f"muestra_{numero:03d}")
        if not os.path.exists(muestra_path):
            return muestra_path
        numero += 1


def mostrar_cuenta_regresiva(cap, senna, segundos=3):
    for restante in range(segundos, 0, -1):
        inicio = time.time()
        while time.time() - inicio < 1:
            ret, frame = cap.read()
            if not ret:
                return False

            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"Senna dinamica: {senna.upper()}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Grabando en {restante}...",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 165, 255),
                3,
            )
            cv2.imshow("Capturador de Senas Dinamicas", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                return False

    return True


def grabar_muestra_dinamica(cap, muestra_path, senna, duracion_segundos):
    crear_carpeta_si_no_existe(muestra_path)

    frame_count = 0
    inicio = time.time()

    while time.time() - inicio < duracion_segundos:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        filename = os.path.join(muestra_path, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(filename, frame)

        cv2.putText(
            frame,
            f"Senna dinamica: {senna.upper()}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Grabando...",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            frame,
            f"Frames: {frame_count}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Capturador de Senas Dinamicas", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    return frame_count


def mostrar_guardado(cap, muestra_path, frame_count):
    inicio = time.time()
    while time.time() - inicio < 1.5:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            "Guardado",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Frames guardados: {frame_count}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Capturador de Senas Dinamicas", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print(f"Guardado: {muestra_path}")
    print(f"Frames guardados: {frame_count}")


def capturar_sena_dinamica():
    crear_carpeta_si_no_existe(dataset_dinamico_path)

    senna = input("Que sena dinamica quieres capturar? (ejemplo: hola_movimiento): ")
    senna = nombre_seguro(senna)

    if not senna:
        print("Debes escribir el nombre de la sena")
        return

    duracion_texto = input("Duracion de cada muestra en segundos (Enter = 3): ").strip()
    try:
        duracion_segundos = float(duracion_texto) if duracion_texto else 3.0
    except ValueError:
        duracion_segundos = 3.0

    senna_path = os.path.join(dataset_dinamico_path, senna)
    crear_carpeta_si_no_existe(senna_path)

    cap = abrir_camara()
    if cap is None:
        return

    print("\nPresiona 'ESPACIO' para grabar una muestra dinamica")
    print("Presiona 'ESC' para salir")
    print(f"Duracion por muestra: {duracion_segundos} segundos")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"Senna dinamica: {senna.upper()}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "ESPACIO: grabar muestra | ESC: salir",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Capturador de Senas Dinamicas", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # ESPACIO
                muestra_path = siguiente_muestra_path(senna_path)

                if not mostrar_cuenta_regresiva(cap, senna):
                    break

                frame_count = grabar_muestra_dinamica(
                    cap,
                    muestra_path,
                    senna,
                    duracion_segundos,
                )
                mostrar_guardado(cap, muestra_path, frame_count)

            elif key == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def mostrar_menu():
    while True:
        print("\n=== SignVoice Colombia - Captura de Dataset ===")
        print("1 -> Capturar sena estatica")
        print("2 -> Capturar sena dinamica")
        print("3 -> Salir")

        opcion = input("Selecciona una opcion: ").strip()

        if opcion == "1":
            capturar_sena_estatica()
        elif opcion == "2":
            capturar_sena_dinamica()
        elif opcion == "3":
            print("Saliendo...")
            break
        else:
            print("Opcion no valida")


if __name__ == "__main__":
    args = parse_args()
    camera_index = args.camara
    mostrar_menu()
