import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
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
HAND_LANDMARKER_TASK = ANDROID_ASSETS_DIR / "hand_landmarker.task"

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
HAND_CROP_PADDING_RATIO = 0.65
MIN_HAND_BOX_SIDE_RATIO = 0.06
MIN_HAND_BOX_AREA_RATIO = 0.0025
BATCH_SIZE = 32
EPOCHS = 45
SEED = 42
MODEL_SEED = 2026
TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15
MIN_SAMPLES_WARNING = 80
USE_DATA_AUGMENTATION = True
USE_CLASS_WEIGHTS = True
ARCHITECTURE_NAME = "cnn_bn_mediapipe_handcrop"
ARCHITECTURE_NOTE = (
    "CNN liviana con BatchNormalization, recorte de mano con MediaPipe "
    "y filtrado de imagenes sin mano"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALPHABET_CLASS_ORDER = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "nn",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


@dataclass
class PreprocessingConfig:
    crop_mode: str
    keep_without_hand: bool
    hand_model_path: Path
    hand_confidence: float


@dataclass
class PreprocessingStats:
    raw_counts: dict
    used_counts: dict
    hand_crop_counts: dict
    center_crop_counts: dict
    skipped_no_hand_counts: dict

    def to_metadata(self):
        return {
            "raw_counts": self.raw_counts,
            "used_counts": self.used_counts,
            "hand_crop_counts": self.hand_crop_counts,
            "center_crop_counts": self.center_crop_counts,
            "skipped_no_hand_counts": self.skipped_no_hand_counts,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo TFLite estatico para el alfabeto LSC.",
    )
    parser.add_argument(
        "--modo",
        choices=["alfabeto", "todas"],
        default="alfabeto",
        help=(
            "alfabeto: usa solo carpetas A-Z/NN. "
            "todas: usa tambien numeros y palabras estaticas."
        ),
    )
    parser.add_argument(
        "--sin-aumento",
        action="store_true",
        help="Desactiva data augmentation durante entrenamiento.",
    )
    parser.add_argument(
        "--sin-pesos-clase",
        action="store_true",
        help="Desactiva pesos de clase balanceados.",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=EPOCHS,
        help=f"Cantidad maxima de epocas de entrenamiento. Por defecto: {EPOCHS}.",
    )
    parser.add_argument(
        "--solo-exportar",
        action="store_true",
        help=(
            "Carga los mejores pesos existentes, evalua y exporta TFLite "
            "sin continuar entrenando."
        ),
    )
    parser.add_argument(
        "--recorte-mano",
        choices=["mediapipe", "centro"],
        default="mediapipe",
        help=(
            "mediapipe: detecta y recorta la mano como la app Android. "
            "centro: usa el recorte central anterior."
        ),
    )
    parser.add_argument(
        "--mantener-sin-mano",
        action="store_true",
        help=(
            "Si MediaPipe no detecta mano, conserva la imagen con recorte central. "
            "Por defecto esas imagenes se omiten para no ensuciar el entrenamiento."
        ),
    )
    parser.add_argument(
        "--modelo-mano",
        default=str(HAND_LANDMARKER_TASK),
        help="Ruta al archivo hand_landmarker.task usado durante el preprocesamiento.",
    )
    parser.add_argument(
        "--confianza-mano",
        type=float,
        default=0.5,
        help="Confianza minima del detector de mano de MediaPipe. Por defecto: 0.5.",
    )
    parser.add_argument(
        "--solo-auditar",
        action="store_true",
        help=(
            "Audita dataset y preprocesamiento con MediaPipe, pero no entrena "
            "ni sobrescribe el modelo."
        ),
    )
    parser.add_argument(
        "--limite-por-clase",
        type=int,
        default=None,
        help=(
            "Usa como maximo esta cantidad de imagenes por clase. Util para "
            "probar auditorias o entrenamientos rapidos sin tocar todo el dataset."
        ),
    )
    return parser.parse_args()


def normalizar_nombre_clase(nombre):
    """Genera un nombre de clase seguro sin modificar la carpeta original."""
    nombre = nombre.strip().lower()
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = re.sub(r"[^a-z0-9_-]", "", nombre)
    return nombre


def auditar_dataset(modo):
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_DIR}")

    clases = []
    carpetas_por_clase = {}
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
        carpetas_por_clase[nombre_normalizado] = carpeta

        if nombre_normalizado != carpeta.name:
            print(
                "ADVERTENCIA: la clase "
                f"'{carpeta.name}' se usara como '{nombre_normalizado}' en etiquetas."
            )

    if modo == "alfabeto":
        faltantes = [
            class_name
            for class_name in ALPHABET_CLASS_ORDER
            if class_name not in carpetas_por_clase
        ]
        if faltantes:
            raise ValueError(
                "Faltan carpetas requeridas para entrenar el alfabeto: "
                + ", ".join(faltantes)
            )

        extras = sorted(set(carpetas_por_clase) - set(ALPHABET_CLASS_ORDER))
        if extras:
            print(
                "Modo alfabeto: se ignoraran clases no alfabeticas: "
                + ", ".join(extras)
            )

        carpetas_a_usar = [
            (class_name, carpetas_por_clase[class_name])
            for class_name in ALPHABET_CLASS_ORDER
        ]
    else:
        carpetas_a_usar = sorted(carpetas_por_clase.items())

    print(f"Modo de entrenamiento: {modo}")

    for nombre_normalizado, carpeta in carpetas_a_usar:
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


def limitar_clases(clases, limite_por_clase):
    if limite_por_clase is None:
        return clases
    if limite_por_clase < 3:
        raise ValueError("--limite-por-clase debe ser al menos 3.")

    print(f"\nModo de prueba: usando maximo {limite_por_clase} imagenes por clase.")
    return [
        (class_name, image_paths[:limite_por_clase])
        for class_name, image_paths in clases
    ]


def cargar_mediapipe_tasks(modelo_path, confianza):
    modelo_path = Path(modelo_path)
    if not modelo_path.exists():
        raise FileNotFoundError(
            "No existe el detector de mano de MediaPipe: "
            f"{modelo_path}. Ejecuta primero la app Android o descarga "
            "hand_landmarker.task en assets."
        )

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise ImportError(
            "Falta MediaPipe para entrenar con recorte de mano. "
            "Instala con: pip install mediapipe"
        ) from error

    base_options = python.BaseOptions(model_asset_path=str(modelo_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=confianza,
        min_hand_presence_confidence=confianza,
        min_tracking_confidence=confianza,
    )
    return vision.HandLandmarker.create_from_options(options), mp


def recortar_centro(imagen_bgr):
    height, width = imagen_bgr.shape[:2]
    roi_size = int(min(height, width) * ROI_RATIO)
    x1 = (width - roi_size) // 2
    y1 = (height - roi_size) // 2
    return imagen_bgr[y1 : y1 + roi_size, x1 : x1 + roi_size]


def caja_mano_plausible(box_width, box_height, image_width, image_height):
    min_image_side = min(image_width, image_height)
    longest_box_side = max(box_width, box_height)
    box_area_ratio = (box_width * box_height) / max(image_width * image_height, 1)

    too_small = (
        longest_box_side < min_image_side * MIN_HAND_BOX_SIDE_RATIO
        or box_area_ratio < MIN_HAND_BOX_AREA_RATIO
    )
    return not too_small


def recortar_mano_mediapipe(imagen_bgr, detector, mp_module):
    rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_image = mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB,
        data=rgb,
    )
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    height, width = imagen_bgr.shape[:2]
    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    if max_x <= min_x or max_y <= min_y:
        return None

    box_width = (max_x - min_x) * width
    box_height = (max_y - min_y) * height
    if not caja_mano_plausible(box_width, box_height, width, height):
        return None

    padding = max(box_width, box_height) * HAND_CROP_PADDING_RATIO
    center_x = ((min_x + max_x) * 0.5) * width
    center_y = ((min_y + max_y) * 0.5) * height
    side = max(box_width, box_height) + padding * 2.0

    left = int(max(0, center_x - side * 0.5))
    top = int(max(0, center_y - side * 0.5))
    right = int(min(width, center_x + side * 0.5))
    bottom = int(min(height, center_y + side * 0.5))

    if right <= left or bottom <= top:
        return None

    return imagen_bgr[top:bottom, left:right]


def crear_canal_bordes(imagen):
    """Extrae bordes suaves para ayudar a distinguir dedos y contornos."""
    gradiente_x = cv2.Sobel(imagen, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_32F, 0, 1, ksize=3)
    bordes = (np.abs(gradiente_x) + np.abs(gradiente_y)) * 0.5
    return np.clip(bordes, 0, 255).astype(np.float32)


def cargar_imagen(imagen_path, config, detector=None, mp_module=None):
    imagen_bgr = cv2.imread(str(imagen_path), cv2.IMREAD_COLOR)
    if imagen_bgr is None or imagen_bgr.size == 0:
        return None, "corrupta"

    metodo = "centro"
    recorte = None

    if config.crop_mode == "mediapipe":
        recorte = recortar_mano_mediapipe(imagen_bgr, detector, mp_module)
        if recorte is not None:
            metodo = "mediapipe"
        elif not config.keep_without_hand:
            return None, "sin_mano"

    if recorte is None:
        recorte = recortar_centro(imagen_bgr)
        metodo = "centro"

    imagen = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    imagen = cv2.resize(imagen, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    imagen = imagen.astype(np.float32)
    bordes = crear_canal_bordes(imagen)
    return np.stack([imagen, bordes], axis=-1), metodo


def cargar_dataset(clases, config, permitir_insuficientes=False):
    imagenes = []
    etiquetas = []
    class_names = [nombre for nombre, _ in clases]
    raw_counts = {nombre: len(image_paths) for nombre, image_paths in clases}
    used_counts = {nombre: 0 for nombre, _ in clases}
    hand_crop_counts = {nombre: 0 for nombre, _ in clases}
    center_crop_counts = {nombre: 0 for nombre, _ in clases}
    skipped_no_hand_counts = {nombre: 0 for nombre, _ in clases}

    detector = None
    mp_module = None
    if config.crop_mode == "mediapipe":
        print("\nCargando MediaPipe HandLandmarker para preprocesar dataset...")
        detector, mp_module = cargar_mediapipe_tasks(
            config.hand_model_path,
            config.hand_confidence,
        )
        print(f"Detector de mano cargado: {config.hand_model_path}")

    try:
        print("\nPreprocesando imagenes...")
        for label_index, (class_name, image_paths) in enumerate(clases):
            for image_path in image_paths:
                imagen, metodo = cargar_imagen(image_path, config, detector, mp_module)
                if imagen is None:
                    if metodo == "sin_mano":
                        skipped_no_hand_counts[class_name] += 1
                    continue

                imagenes.append(imagen)
                etiquetas.append(label_index)
                used_counts[class_name] += 1
                if metodo == "mediapipe":
                    hand_crop_counts[class_name] += 1
                else:
                    center_crop_counts[class_name] += 1

            print(
                f"- {class_name}: usadas={used_counts[class_name]}, "
                f"mano={hand_crop_counts[class_name]}, "
                f"centro={center_crop_counts[class_name]}, "
                f"omitidas_sin_mano={skipped_no_hand_counts[class_name]}"
            )
            if used_counts[class_name] < MIN_SAMPLES_WARNING:
                print(
                    "  ADVERTENCIA: pocas muestras utiles despues del recorte. "
                    f"Recomendado minimo: {MIN_SAMPLES_WARNING}"
                )
    finally:
        if detector is not None:
            detector.close()

    clases_insuficientes = [
        class_name
        for class_name, cantidad in used_counts.items()
        if cantidad < 3
    ]
    if clases_insuficientes and not permitir_insuficientes:
        raise ValueError(
            "Despues del preprocesamiento quedaron clases con menos de 3 imagenes: "
            + ", ".join(clases_insuficientes)
        )
    if clases_insuficientes:
        print(
            "\nADVERTENCIA: clases con menos de 3 imagenes despues del "
            "preprocesamiento: " + ", ".join(clases_insuficientes)
        )

    if not imagenes:
        raise ValueError("No quedaron imagenes validas despues del preprocesamiento.")

    x = np.asarray(imagenes, dtype=np.float32)
    y = np.asarray(etiquetas, dtype=np.int64)
    stats = PreprocessingStats(
        raw_counts=raw_counts,
        used_counts=used_counts,
        hand_crop_counts=hand_crop_counts,
        center_crop_counts=center_crop_counts,
        skipped_no_hand_counts=skipped_no_hand_counts,
    )

    return x, y, class_names, stats


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
            tf.keras.layers.Conv2D(24, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(48, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.SeparableConv2D(96, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.SeparableConv2D(128, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.SpatialDropout2D(0.12),
            tf.keras.layers.SeparableConv2D(160, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes),
        ],
        name="signvoice_alfabeto_cnn_mediapipe",
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


def ruta_para_metadata(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path)


def exportar_modelo_tflite(
    model,
    class_names,
    test_accuracy,
    test_loss,
    modo,
    usar_aumento,
    usar_pesos_clase,
    preprocessing_config,
    preprocessing_stats,
):
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
        "labels_file_order": [class_name.upper() for class_name in class_names],
        "dataset_mode": modo,
        "split_seed": SEED,
        "model_seed": MODEL_SEED,
        "architecture": ARCHITECTURE_NAME,
        "architecture_note": ARCHITECTURE_NOTE,
        "data_augmentation": usar_aumento,
        "class_weights": usar_pesos_clase,
        "input_value_range": "0-255",
        "normalization": "Rescaling(1/255) inside model",
        "preprocessing": (
            "MediaPipe HandLandmarker hand crop by default, channel 0 = "
            "grayscale luma, channel 1 = Sobel edge magnitude"
        ),
        "crop_mode": preprocessing_config.crop_mode,
        "keep_without_hand": preprocessing_config.keep_without_hand,
        "hand_model": ruta_para_metadata(preprocessing_config.hand_model_path),
        "hand_confidence": preprocessing_config.hand_confidence,
        "center_roi_ratio": ROI_RATIO,
        "hand_crop_padding_ratio": HAND_CROP_PADDING_RATIO,
        "sample_counts": preprocessing_stats.used_counts,
        "preprocessing_stats": preprocessing_stats.to_metadata(),
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
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(MODEL_SEED)

    usar_aumento = USE_DATA_AUGMENTATION and not args.sin_aumento
    usar_pesos_clase = USE_CLASS_WEIGHTS and not args.sin_pesos_clase
    preprocessing_config = PreprocessingConfig(
        crop_mode=args.recorte_mano,
        keep_without_hand=args.mantener_sin_mano,
        hand_model_path=Path(args.modelo_mano),
        hand_confidence=args.confianza_mano,
    )

    clases = limitar_clases(auditar_dataset(args.modo), args.limite_por_clase)
    x, y, class_names, preprocessing_stats = cargar_dataset(
        clases,
        preprocessing_config,
        permitir_insuficientes=args.solo_auditar,
    )

    if args.solo_auditar:
        print("\nAuditoria finalizada. No se entreno ni se sobrescribio el modelo.")
        print(f"Total de imagenes usadas: {len(x)}")
        print(
            "Total omitidas sin mano: "
            f"{sum(preprocessing_stats.skipped_no_hand_counts.values())}"
        )
        return

    x_train, y_train, x_val, y_val, x_test, y_test = dividir_dataset(x, y)

    print("\nResumen de division:")
    print(f"- Entrenamiento: {len(x_train)} imagenes")
    print(f"- Validacion: {len(x_val)} imagenes")
    print(f"- Prueba: {len(x_test)} imagenes")
    print(f"- Clases: {class_names}")
    print(f"- Data augmentation: {'si' if usar_aumento else 'no'}")
    print(f"- Pesos de clase: {'si' if usar_pesos_clase else 'no'}")
    print(f"- Recorte de mano: {preprocessing_config.crop_mode}")
    print(f"- Imagenes usadas: {len(x)}")
    print(
        "- Imagenes omitidas sin mano: "
        f"{sum(preprocessing_stats.skipped_no_hand_counts.values())}"
    )

    class_weight = calcular_pesos_clase(y_train) if usar_pesos_clase else None

    model = crear_modelo(num_classes=len(class_names))
    model.summary()

    if args.solo_exportar:
        if not BEST_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"No existen pesos para exportar: {BEST_WEIGHTS_PATH}"
            )
        print(f"\nCargando pesos existentes: {BEST_WEIGHTS_PATH}")
        model.load_weights(BEST_WEIGHTS_PATH)
    else:
        print("\nEntrenando modelo mejorado...")
        if usar_aumento:
            train_ds, val_ds, _ = crear_datasets_tf(x_train, y_train, x_val, y_val, x_test, y_test)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epocas,
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
                epochs=args.epocas,
                callbacks=crear_callbacks(),
                class_weight=class_weight,
                shuffle=True,
                verbose=1,
            )

    if BEST_WEIGHTS_PATH.exists() and not args.solo_exportar:
        model.load_weights(BEST_WEIGHTS_PATH)

    model.save(str(BEST_MODEL_PATH), include_optimizer=False)
    test_accuracy, test_loss = guardar_metricas(model, x_test, y_test, class_names)
    exportar_modelo_tflite(
        model,
        class_names,
        test_accuracy,
        test_loss,
        args.modo,
        usar_aumento,
        usar_pesos_clase,
        preprocessing_config,
        preprocessing_stats,
    )


if __name__ == "__main__":
    main()
