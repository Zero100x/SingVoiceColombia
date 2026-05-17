import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT
DATASET_DINAMICO_DIR = PROJECT_ROOT / "datasets" / "dynamic_signs"
MODELOS_DIR = PROJECT_ROOT / "ai" / "models"
ANDROID_ASSETS_DIR = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets"

BEST_MODEL_PATH = MODELOS_DIR / "modelo_dinamico_mejor.h5"
BEST_WEIGHTS_PATH = MODELOS_DIR / "modelo_dinamico_mejor.weights.h5"
SAVED_MODEL_DIR = MODELOS_DIR / "modelo_dinamico_savedmodel"
TFLITE_OUTPUT = ANDROID_ASSETS_DIR / "modelo_dinamico.tflite"
LABELS_OUTPUT = ANDROID_ASSETS_DIR / "dynamic_labels.txt"
METADATA_OUTPUT = MODELOS_DIR / "modelo_dinamico_metadata.json"
CONFUSION_MATRIX_OUTPUT = MODELOS_DIR / "matriz_confusion_dinamico.csv"
CLASSIFICATION_REPORT_OUTPUT = MODELOS_DIR / "reporte_clasificacion_dinamico.txt"
TRAINING_HISTORY_OUTPUT = MODELOS_DIR / "historial_entrenamiento_dinamico.csv"

IMAGE_SIZE = (96, 96)
SEQUENCE_LENGTH = 16
CHANNELS = 6
ROI_RATIO = 0.75
WINDOWS_PER_VIDEO_SAMPLE = 3
BATCH_SIZE = 32
EPOCHS = 40
SPLIT_SEED = 42
MODEL_SEED = 2026
TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15
MIN_FRAMES_PER_SAMPLE = 8
MIN_SAMPLES_TO_TRAIN = 6
MIN_SAMPLES_WARNING = 30
MIN_TRAIN_MOTION_SCORE = 2.0
CONFIDENCE_THRESHOLD = 0.65
USE_CLASS_WEIGHTS = True
USE_UNKNOWN_CLASS = True
UNKNOWN_CLASS_NAME = "desconocido"
MIN_UNKNOWN_SAMPLES_TO_TRAIN = 3
ARCHITECTURE_NAME = "motion_summary_cnn_from_dynamic_videos"
ARCHITECTURE_NOTE = (
    "Resume cada video en 6 canales: frame inicial, bordes iniciales, frame final, "
    "bordes finales, mapa de movimiento y bordes del movimiento."
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SKIPPED_DYNAMIC_CLASSES = []
LOW_MOTION_WINDOWS_SKIPPED = 0


def normalizar_nombre_clase(nombre):
    """Genera etiquetas seguras sin cambiar la carpeta original."""
    nombre = nombre.strip().lower()
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = re.sub(r"[^a-z0-9_-]", "", nombre)
    return nombre


def auditar_dataset_dinamico():
    global SKIPPED_DYNAMIC_CLASSES

    if not DATASET_DINAMICO_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset dinamico: {DATASET_DINAMICO_DIR}")

    clases = []
    nombres_normalizados = set()
    print("Auditando dataset dinamico...")

    for carpeta_clase in sorted(path for path in DATASET_DINAMICO_DIR.iterdir() if path.is_dir()):
        nombre_normalizado = normalizar_nombre_clase(carpeta_clase.name)
        if not nombre_normalizado:
            print(f"ADVERTENCIA: carpeta ignorada por nombre invalido: {carpeta_clase.name}")
            continue

        if nombre_normalizado in nombres_normalizados:
            raise ValueError(
                "Hay clases repetidas despues de normalizar nombres. "
                f"Revisa la carpeta: {carpeta_clase.name}"
            )

        nombres_normalizados.add(nombre_normalizado)
        muestras_validas = []
        muestras_invalidas = 0
        frames_corruptos = 0

        for carpeta_muestra in sorted(path for path in carpeta_clase.iterdir() if path.is_dir()):
            frames_validos = []

            for frame_path in sorted(carpeta_muestra.iterdir()):
                if frame_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                if frame is None or frame.size == 0:
                    frames_corruptos += 1
                    continue

                frames_validos.append(frame_path)

            if len(frames_validos) < MIN_FRAMES_PER_SAMPLE:
                muestras_invalidas += 1
                continue

            muestras_validas.append(frames_validos)

        print(f"- {nombre_normalizado}: {len(muestras_validas)} muestras validas")

        if muestras_invalidas:
            print(
                "  ADVERTENCIA: "
                f"{muestras_invalidas} muestras ignoradas por tener pocos frames"
            )

        if frames_corruptos:
            print(f"  ADVERTENCIA: {frames_corruptos} frames corruptos ignorados")

        if len(muestras_validas) < MIN_SAMPLES_WARNING:
            print(
                "  ADVERTENCIA: pocas muestras para movimiento. "
                f"Recomendado minimo: {MIN_SAMPLES_WARNING}"
            )

        clases.append((nombre_normalizado, muestras_validas))

    clases = [(nombre, muestras) for nombre, muestras in clases if muestras]
    clases_con_pocas_muestras = [
        (nombre, muestras)
        for nombre, muestras in clases
        if len(muestras) < MIN_SAMPLES_TO_TRAIN
    ]
    SKIPPED_DYNAMIC_CLASSES = [
        {"class": nombre, "samples": len(muestras)}
        for nombre, muestras in clases_con_pocas_muestras
    ]
    clases_entrenables = [
        (nombre, muestras)
        for nombre, muestras in clases
        if len(muestras) >= MIN_SAMPLES_TO_TRAIN
    ]

    if USE_UNKNOWN_CLASS:
        muestras_desconocidas = [
            muestra
            for _, muestras in clases_con_pocas_muestras
            for muestra in muestras
        ]
        if len(muestras_desconocidas) >= MIN_UNKNOWN_SAMPLES_TO_TRAIN:
            clases_entrenables.append((UNKNOWN_CLASS_NAME, muestras_desconocidas))

    clases = clases_entrenables

    if SKIPPED_DYNAMIC_CLASSES:
        print("\nClases omitidas por pocas muestras:")
        for item in SKIPPED_DYNAMIC_CLASSES:
            print(
                f"- {item['class']}: {item['samples']} muestras "
                f"(minimo tecnico: {MIN_SAMPLES_TO_TRAIN})"
            )
        if USE_UNKNOWN_CLASS:
            total_unknown = sum(item["samples"] for item in SKIPPED_DYNAMIC_CLASSES)
            if total_unknown >= MIN_UNKNOWN_SAMPLES_TO_TRAIN:
                print(
                    f"- {total_unknown} muestras se usaran como '{UNKNOWN_CLASS_NAME}' "
                    "para evitar falsas predicciones."
                )

    if len(clases) < 2:
        raise ValueError("Necesitas al menos 2 senas dinamicas con muestras validas.")

    validar_minimo_entrenamiento(clases)
    return clases


def validar_minimo_entrenamiento(clases):
    clases_con_pocas_muestras = [
        f"{nombre} ({len(muestras)})"
        for nombre, muestras in clases
        if len(muestras) < MIN_SAMPLES_TO_TRAIN
    ]

    if clases_con_pocas_muestras:
        raise ValueError(
            "Todavia no hay suficientes muestras para entrenar un modelo dinamico.\n"
            f"Minimo tecnico por clase: {MIN_SAMPLES_TO_TRAIN}\n"
            "Clases con pocas muestras: "
            + ", ".join(clases_con_pocas_muestras)
            + "\nCaptura mas muestras con: python capturar_dataset.py -> opcion 2"
        )


def crear_canal_bordes(imagen):
    gradiente_x = cv2.Sobel(imagen, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_32F, 0, 1, ksize=3)
    bordes = (np.abs(gradiente_x) + np.abs(gradiente_y)) * 0.5
    return np.clip(bordes, 0, 255).astype(np.float32)


def cargar_frame(frame_path):
    imagen = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    height, width = imagen.shape[:2]
    roi_size = int(min(height, width) * ROI_RATIO)
    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2
    imagen = imagen[y1 : y1 + roi_size, x1 : x1 + roi_size]
    imagen = cv2.resize(imagen, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return imagen.astype(np.float32)


def seleccionar_frames_secuencia(frames):
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH)
    indices = np.round(indices).astype(int)
    return [frames[indice] for indice in indices]


def generar_ventanas_muestra(frames):
    """Crea ventanas temporales sin mezclar videos entre train/val/test."""
    if len(frames) <= SEQUENCE_LENGTH:
        return [frames]

    max_start = len(frames) - SEQUENCE_LENGTH
    starts = np.linspace(0, max_start, WINDOWS_PER_VIDEO_SAMPLE)
    starts = np.round(starts).astype(int)
    return [frames[start : start + SEQUENCE_LENGTH] for start in starts]


def crear_resumen_movimiento(frames):
    secuencia = [cargar_frame(frame_path) for frame_path in seleccionar_frames_secuencia(frames)]
    primer_frame = secuencia[0]
    ultimo_frame = secuencia[-1]
    diferencias = [
        np.abs(secuencia[indice + 1] - secuencia[indice])
        for indice in range(len(secuencia) - 1)
    ]
    movimiento = np.mean(diferencias, axis=0).astype(np.float32)

    resumen = np.stack(
        [
            primer_frame,
            crear_canal_bordes(primer_frame),
            ultimo_frame,
            crear_canal_bordes(ultimo_frame),
            movimiento,
            crear_canal_bordes(movimiento),
        ],
        axis=-1,
    ).astype(np.float32)
    return resumen, float(movimiento.mean())


def cargar_resumen_movimiento(frames):
    resumen, _ = crear_resumen_movimiento(frames)
    return resumen


def dividir_muestras(clases):
    train_samples = []
    val_samples = []
    test_samples = []
    class_names = [nombre for nombre, _ in clases]

    for label_index, (_, muestras_clase) in enumerate(clases):
        train_val, test = train_test_split(
            muestras_clase,
            test_size=TEST_SPLIT,
            random_state=SPLIT_SEED,
            shuffle=True,
        )

        val_fraction = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
        train, val = train_test_split(
            train_val,
            test_size=val_fraction,
            random_state=SPLIT_SEED,
            shuffle=True,
        )

        train_samples.extend((label_index, frames) for frames in train)
        val_samples.extend((label_index, frames) for frames in val)
        test_samples.extend((label_index, frames) for frames in test)

    return train_samples, val_samples, test_samples, class_names


def cargar_dataset_desde_muestras(samples):
    global LOW_MOTION_WINDOWS_SKIPPED

    resumenes = []
    etiquetas = []

    for label_index, frames in samples:
        for ventana in generar_ventanas_muestra(frames):
            resumen, motion_score = crear_resumen_movimiento(ventana)
            if motion_score < MIN_TRAIN_MOTION_SCORE:
                LOW_MOTION_WINDOWS_SKIPPED += 1
                continue

            resumenes.append(resumen)
            etiquetas.append(label_index)

    x = np.asarray(resumenes, dtype=np.float32)
    y = np.asarray(etiquetas, dtype=np.int64)
    return x, y


def calcular_pesos_clase(y_train):
    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
    return {int(clase): float(peso) for clase, peso in zip(clases, pesos)}


def crear_modelo(num_classes):
    """CNN liviana para resumenes compactos de movimiento."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes),
        ],
        name="signvoice_dinamico_motion_summary_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def crear_callbacks():
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_WEIGHTS_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(TRAINING_HISTORY_OUTPUT)),
    ]


def guardar_metricas(model, x_test, y_test, class_names):
    print("\nEvaluando mejor modelo dinamico en prueba...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

    logits = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    matriz = confusion_matrix(y_test, y_pred)
    np.savetxt(CONFUSION_MATRIX_OUTPUT, matriz, fmt="%d", delimiter=",")

    reporte = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    CLASSIFICATION_REPORT_OUTPUT.write_text(reporte, encoding="utf-8")

    print("\nReporte de clasificacion dinamico:")
    print(reporte)
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_OUTPUT}")
    print(f"Reporte guardado en: {CLASSIFICATION_REPORT_OUTPUT}")

    return float(test_accuracy), float(test_loss)


def exportar_modelo_tflite(model, class_names, test_accuracy, test_loss):
    if SAVED_MODEL_DIR.exists():
        shutil.rmtree(SAVED_MODEL_DIR)

    print("\nExportando modelo dinamico intermedio...")
    model.export(SAVED_MODEL_DIR)

    print("Convirtiendo modelo dinamico a TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TFLITE_OUTPUT.write_bytes(tflite_model)
    LABELS_OUTPUT.write_text(
        "\n".join(class_name.upper() for class_name in class_names) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "image_size": list(IMAGE_SIZE),
        "sequence_length": SEQUENCE_LENGTH,
        "channels": CHANNELS,
        "windows_per_video_sample": WINDOWS_PER_VIDEO_SAMPLE,
        "classes": class_names,
        "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED,
        "architecture": ARCHITECTURE_NAME,
        "architecture_note": ARCHITECTURE_NOTE,
        "input_value_range": "0-255",
        "normalization": "Rescaling(1/255) inside model",
        "preprocessing": (
            "center ROI crop 0.75, channels = first grayscale, first edges, "
            "last grayscale, last edges, mean absolute motion, motion edges"
        ),
        "roi_ratio": ROI_RATIO,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_train_motion_score": MIN_TRAIN_MOTION_SCORE,
        "use_unknown_class": USE_UNKNOWN_CLASS,
        "unknown_class_name": UNKNOWN_CLASS_NAME,
        "min_samples_to_train": MIN_SAMPLES_TO_TRAIN,
        "skipped_classes": SKIPPED_DYNAMIC_CLASSES,
        "low_motion_windows_skipped": LOW_MOTION_WINDOWS_SKIPPED,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "tflite_model": str(TFLITE_OUTPUT.relative_to(BASE_DIR)),
    }
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Modelo Keras dinamico guardado en: {BEST_MODEL_PATH}")
    print(f"Mejores pesos dinamicos guardados en: {BEST_WEIGHTS_PATH}")
    print(f"Modelo TFLite dinamico guardado en: {TFLITE_OUTPUT}")
    print(f"Etiquetas dinamicas guardadas en: {LABELS_OUTPUT}")
    print(f"Metadata dinamica guardada en: {METADATA_OUTPUT}")


def main():
    global LOW_MOTION_WINDOWS_SKIPPED

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(MODEL_SEED)
    LOW_MOTION_WINDOWS_SKIPPED = 0

    clases = auditar_dataset_dinamico()
    train_samples, val_samples, test_samples, class_names = dividir_muestras(clases)
    x_train, y_train = cargar_dataset_desde_muestras(train_samples)
    x_val, y_val = cargar_dataset_desde_muestras(val_samples)
    x_test, y_test = cargar_dataset_desde_muestras(test_samples)

    print("\nResumen de division dinamica:")
    print(f"- Entrenamiento: {len(train_samples)} videos / {len(x_train)} ventanas")
    print(f"- Validacion: {len(val_samples)} videos / {len(x_val)} ventanas")
    print(f"- Prueba: {len(test_samples)} videos / {len(x_test)} ventanas")
    print(f"- Ventanas omitidas por poco movimiento: {LOW_MOTION_WINDOWS_SKIPPED}")
    print(f"- Clases: {class_names}")
    print(f"- Entrada del modelo: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x{CHANNELS}")

    model = crear_modelo(num_classes=len(class_names))
    model.summary()
    class_weight = calcular_pesos_clase(y_train) if USE_CLASS_WEIGHTS else None

    print("\nEntrenando modelo dinamico...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=crear_callbacks(),
        class_weight=class_weight,
        shuffle=True,
        verbose=1,
    )

    if BEST_WEIGHTS_PATH.exists():
        model.load_weights(BEST_WEIGHTS_PATH)

    model.save(str(BEST_MODEL_PATH), include_optimizer=False)
    test_accuracy, test_loss = guardar_metricas(model, x_test, y_test, class_names)
    exportar_modelo_tflite(model, class_names, test_accuracy, test_loss)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
