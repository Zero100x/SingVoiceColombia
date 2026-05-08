import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from skimage.feature import hog
from skimage import io, exposure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ai" / "models"
DATASET_DIR = PROJECT_ROOT / "datasets" / "static_signs"
data_file = MODELS_DIR / "datos_caracteristicas.pkl"
labels_file = MODELS_DIR / "etiquetas.pkl"
dataset_path = DATASET_DIR

# Listas para guardar datos
all_data = []
all_labels = []

def extraer_caracteristicas(imagen):
    """Extrae características de HOG + color de uma imagen"""
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar para consistencia
    gray = cv2.resize(gray, (128, 128))
    
    # Extraer HOG (Histogram of Oriented Gradients)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=False)
    
    # Extraer características de color (histograma HSV)
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
    
    # Normalizar histogramas
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    # Combinar todas las características
    features = np.concatenate([hog_features, hist_h, hist_s, hist_v])
    
    return features

# Carpetas de señas
senna_folders = [f for f in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, f))]

print(f"Procesando {len(senna_folders)} señas...")

# Procesar cada seña
for idx, senna in enumerate(senna_folders):
    senna_path = os.path.join(dataset_path, senna)
    images = [f for f in os.listdir(senna_path) if f.endswith(('.jpg', '.png'))]
    
    print(f"\n📸 Procesando '{senna}' ({idx+1}/{len(senna_folders)})")
    print(f"   Imágenes encontradas: {len(images)}")
    
    for img_file in images:
        img_path = os.path.join(senna_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        try:
            # Extraer características
            features = extraer_caracteristicas(image)
            
            all_data.append(features)
            all_labels.append(senna)
            print(f"   ✓ {img_file} - Características extraídas")
        except Exception as e:
            print(f"   ✗ {img_file} - Error: {e}")

print(f"\n✓ Procesamiento completado!")
print(f"✓ Total de muestras: {len(all_data)}")
print(f"✓ Señas distintas: {len(set(all_labels))}")

# Guardar datos
if len(all_data) > 0:
    # Convertir a numpy
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar
    with open(data_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    with open(labels_file, 'wb') as f:
        pickle.dump(all_labels, f)
    
    print(f"\n✓ Datos guardados en: {data_file}")
    print(f"✓ Etiquetas guardadas en: {labels_file}")
    print(f"\nINFORMACIÓN DEL DATASET:")
    print(f"- Forma de datos: {all_data.shape}")
    print(f"- Señas: {list(set(all_labels))}")
    
    # Contar imágenes por seña
    for senna in set(all_labels):
        count = len([x for x in all_labels if x == senna])
        print(f"  • {senna}: {count} muestras")
else:
    print("❌ No se pudo extraer datos. Verifica que haya imágenes en el dataset.")
