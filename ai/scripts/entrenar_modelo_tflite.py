import json
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT
DATASET_DIR = PROJECT_ROOT / "datasets" / "static_signs"
MODELOS_DIR = PROJECT_ROOT / "ai" / "models"
ANDROID_ASSETS_DIR = PROJECT_ROOT / "frontend" / "android" / "app" / "src" / "main" / "assets"

BEST_MODEL_PATH = MODELOS_DIR / "modelo_alfabeto_mejor.h5"
BEST_WEIGHTS_PATH = MODELOS_DIR / "modelo_alfabeto_mejor.weights.h5"
SAVED_MODEL_DIR = MODELOS_DIR / "modelo_alfabeto_savedmodel"
TFLITE_OUTPUT = ANDROID_ASSETS_DIR / "modelo_alfabeto.tflite"
LABELS_OUTPUT = ANDROID_ASSETS_DIR / "labels.txt"
METADATA_OUTPUT = MODELOS_DIR / "modelo_alfabeto_metadata.json"
CONFUSION_MATRIX_OUTPUT = MODELOS_DIR / "matriz_confusion_alfabeto.csv"
CLASSIFICATION_REPORT_OUTPUT = MODELOS_DIR / "reporte_clasificacion_alfabeto.txt"
TRAINING_HISTORY_OUTPUT = MODELOS_DIR / "historial_entrenamiento_alfabeto.csv"

IMAGE_SIZE = (96, 96)
CHANNELS = 2
ROI_RATIO = 0.75
BATCH_SIZE = 32
EPOCHS = 35
SEED = 42
MODEL_SEED = 2026
TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15
MIN_SAMPLES_WARNING = 80
USE_DATA_AUGMENTATION = False
USE_CLASS_WEIGHTS = False
ARCHITECTURE_NAME = "cnn_extra2_relu_dropout02"
ARCHITECTURE_NOTE = "PDF idea: base CNN + Dropout 0.2 + two extra convolution/pooling layers"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalizar_nombre_clase(nombre):
    """Genera un nombre de clase seguro sin modificar la carpeta original."""
    nombre = nombre.strip().lower()
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = re.sub(r"[^a-z0-9_-]", "", nombre)
    return nombre


def auditar_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_DIR}")

    clases = []
    nombres_normalizados = set()
    total_corruptas = 0

    print("Auditando dataset...")

    for carpeta in sorted(path for path in DATASET_DIR.iterdir() if path.is_dir()):
        nombre_normalizado = normalizar_nombre_clase(carpeta.name)
        if not nombre_normalizado:
            print(f"ADVERTENCIA: carpeta ignorada por nombre invalido: {carpeta.name}")
            continue

        if nombre_normalizado in nombres_normalizados:
            raise ValueError(
                "Hay clases que quedan repetidas despues de normalizar nombres. "
                f"Revisa la carpeta: {carpeta.name}"
            )

        nombres_normalizados.add(nombre_normalizado)

        if nombre_normalizado != carpeta.name:
            print(
                "ADVERTENCIA: la clase "
                f"'{carpeta.name}' se usara como '{nombre_normalizado}' en etiquetas."
            )

        imagenes_validas = []
        corruptas = 0

        for imagen_path in sorted(carpeta.iterdir()):
            if imagen_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            imagen = cv2.imread(str(imagen_path), cv2.IMREAD_GRAYSCALE)
            if imagen is None or imagen.size == 0:
                corruptas += 1
                continue

            imagenes_validas.append(imagen_path)

        total_corruptas += corruptas
        cantidad = len(imagenes_validas)

        print(f"- {nombre_normalizado}: {cantidad} imagenes validas")

        if corruptas:
            print(f"  ADVERTENCIA: {corruptas} imagenes corruptas o vacias ignoradas")

        if cantidad < MIN_SAMPLES_WARNING:
            print(
                "  ADVERTENCIA: pocas muestras. "
                f"Recomendado minimo: {MIN_SAMPLES_WARNING}"
            )

        if cantidad < 3:
            raise ValueError(
                f"La clase '{nombre_normalizado}' necesita al menos 3 imagenes validas."
            )

        clases.append((nombre_normalizado, imagenes_validas))

    if not clases:
        raise ValueError("No se encontraron clases validas para entrenar.")

    if total_corruptas:
        print(f"Total de imagenes corruptas ignoradas: {total_corruptas}")

    return clases


def crear_canal_bordes(imagen):
    """Extrae bordes suaves para ayudar a distinguir dedos y contornos."""
    gradiente_x = cv2.Sobel(imagen, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_32F, 0, 1, ksize=3)
    bordes = (np.abs(gradiente_x) + np.abs(gradiente_y)) * 0.5
    return np.clip(bordes, 0, 255).astype(np.float32)


def cargar_imagen(imagen_path):
    imagen = cv2.imread(str(imagen_path), cv2.IMREAD_GRAYSCALE)
    height, width = imagen.shape[:2]
    roi_size = int(min(height, width) * ROI_RATIO)
    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2
    imagen = imagen[y1 : y1 + roi_size, x1 : x1 + roi_size]
    imagen = cv2.resize(imagen, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    imagen = imagen.astype(np.float32)
    bordes = crear_canal_bordes(imagen)
    return np.stack([imagen, bordes], axis=-1)


def cargar_dataset(clases):
    imagenes = []
    etiquetas = []
    class_names = [nombre for nombre, _ in clases]

    for label_index, (_, image_paths) in enumerate(clases):
        for image_path in image_paths:
            imagenes.append(cargar_imagen(image_path))
            etiquetas.append(label_index)

    x = np.asarray(imagenes, dtype=np.float32)
    y = np.asarray(etiquetas, dtype=np.int64)

    return x, y, class_names


def dividir_dataset(x, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SPLIT,
        random_state=SEED,
        stratify=y,
    )

    val_fraction = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=SEED,
        stratify=y_train_val,
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def crear_datasets_tf(x_train, y_train, x_val, y_val, x_test, y_test):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="aumento_datos_ligero",
    )

    def aplicar_aumento(images, labels):
        images = data_augmentation(images, training=True)
        images = tf.clip_by_value(images, 0.0, 255.0)
        return images, labels

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(aplicar_aumento, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def crear_modelo(num_classes):
    """CNN liviana para Android; la normalizacion queda dentro del modelo."""
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
        name="signvoice_alfabeto_cnn_pdf",
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


def calcular_pesos_clase(y_train):
    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
    return {int(clase): float(peso) for clase, peso in zip(clases, pesos)}


def guardar_metricas(model, x_test, y_test, class_names):
    print("\nEvaluando mejor modelo en prueba...")
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

    print("\nReporte de clasificacion:")
    print(reporte)
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_OUTPUT}")
    print(f"Reporte guardado en: {CLASSIFICATION_REPORT_OUTPUT}")

    return float(test_accuracy), float(test_loss)


def exportar_modelo_tflite(model, class_names, test_accuracy, test_loss):
    if SAVED_MODEL_DIR.exists():
        shutil.rmtree(SAVED_MODEL_DIR)

    print("\nExportando modelo intermedio...")
    model.export(SAVED_MODEL_DIR)

    print("Convirtiendo a TensorFlow Lite...")
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
        "channels": CHANNELS,
        "classes": class_names,
        "split_seed": SEED,
        "model_seed": MODEL_SEED,
        "architecture": ARCHITECTURE_NAME,
        "architecture_note": ARCHITECTURE_NOTE,
        "input_value_range": "0-255",
        "normalization": "Rescaling(1/255) inside model",
        "preprocessing": (
            "center ROI crop 0.75, channel 0 = grayscale luma, "
            "channel 1 = Sobel edge magnitude"
        ),
        "roi_ratio": ROI_RATIO,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "tflite_model": str(TFLITE_OUTPUT.relative_to(BASE_DIR)),
    }
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Modelo Keras guardado en: {BEST_MODEL_PATH}")
    print(f"Mejores pesos guardados en: {BEST_WEIGHTS_PATH}")
    print(f"Modelo TFLite guardado en: {TFLITE_OUTPUT}")
    print(f"Etiquetas guardadas en: {LABELS_OUTPUT}")
    print(f"Metadata guardada en: {METADATA_OUTPUT}")


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(MODEL_SEED)

    clases = auditar_dataset()
    x, y, class_names = cargar_dataset(clases)
    x_train, y_train, x_val, y_val, x_test, y_test = dividir_dataset(x, y)

    print("\nResumen de division:")
    print(f"- Entrenamiento: {len(x_train)} imagenes")
    print(f"- Validacion: {len(x_val)} imagenes")
    print(f"- Prueba: {len(x_test)} imagenes")
    print(f"- Clases: {class_names}")

    class_weight = calcular_pesos_clase(y_train) if USE_CLASS_WEIGHTS else None

    model = crear_modelo(num_classes=len(class_names))
    model.summary()

    print("\nEntrenando modelo mejorado...")
    if USE_DATA_AUGMENTATION:
        train_ds, val_ds, _ = crear_datasets_tf(x_train, y_train, x_val, y_val, x_test, y_test)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=crear_callbacks(),
            class_weight=class_weight,
            verbose=1,
        )
    else:
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
    main()
