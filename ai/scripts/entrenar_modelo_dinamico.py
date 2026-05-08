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
CHANNELS_PER_FRAME = 2
STACKED_CHANNELS = SEQUENCE_LENGTH * CHANNELS_PER_FRAME
BATCH_SIZE = 8
EPOCHS = 40
SPLIT_SEED = 42
MODEL_SEED = 2026
TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15
MIN_FRAMES_PER_SAMPLE = 8
MIN_SAMPLES_TO_TRAIN = 6
MIN_SAMPLES_WARNING = 30
CONFIDENCE_THRESHOLD = 0.65

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalizar_nombre_clase(nombre):
    """Genera etiquetas seguras sin cambiar la carpeta original."""
    nombre = nombre.strip().lower()
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = re.sub(r"[^a-z0-9_-]", "", nombre)
    return nombre


def auditar_dataset_dinamico():
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
    imagen = cv2.resize(imagen, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    imagen = imagen.astype(np.float32)
    bordes = crear_canal_bordes(imagen)
    return imagen, bordes


def cargar_muestra(frames):
    """Convierte una muestra completa en un volumen temporal TFLite-friendly."""
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH)
    indices = np.round(indices).astype(int)

    canales = []
    for indice in indices:
        gris, bordes = cargar_frame(frames[indice])
        canales.append(gris)
        canales.append(bordes)

    return np.stack(canales, axis=-1).astype(np.float32)


def cargar_dataset(clases):
    muestras = []
    etiquetas = []
    class_names = [nombre for nombre, _ in clases]

    for label_index, (_, muestras_clase) in enumerate(clases):
        for frames in muestras_clase:
            muestras.append(cargar_muestra(frames))
            etiquetas.append(label_index)

    x = np.asarray(muestras, dtype=np.float32)
    y = np.asarray(etiquetas, dtype=np.int64)
    return x, y, class_names


def dividir_dataset(x, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SPLIT,
        random_state=SPLIT_SEED,
        stratify=y,
    )

    val_fraction = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=SPLIT_SEED,
        stratify=y_train_val,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def crear_modelo(num_classes):
    """CNN temporal liviana: los frames quedan ordenados como canales."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], STACKED_CHANNELS)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(num_classes),
        ],
        name="signvoice_dinamico_temporal_cnn",
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
            patience=3,
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
        "channels_per_frame": CHANNELS_PER_FRAME,
        "stacked_channels": STACKED_CHANNELS,
        "classes": class_names,
        "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED,
        "input_value_range": "0-255",
        "normalization": "Rescaling(1/255) inside model",
        "preprocessing": "sampled grayscale frames plus Sobel edge channels",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
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
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(MODEL_SEED)

    clases = auditar_dataset_dinamico()
    x, y, class_names = cargar_dataset(clases)
    x_train, y_train, x_val, y_val, x_test, y_test = dividir_dataset(x, y)

    print("\nResumen de division dinamica:")
    print(f"- Entrenamiento: {len(x_train)} muestras")
    print(f"- Validacion: {len(x_val)} muestras")
    print(f"- Prueba: {len(x_test)} muestras")
    print(f"- Clases: {class_names}")
    print(f"- Entrada del modelo: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x{STACKED_CHANNELS}")

    model = crear_modelo(num_classes=len(class_names))
    model.summary()

    print("\nEntrenando modelo dinamico...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=crear_callbacks(),
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
