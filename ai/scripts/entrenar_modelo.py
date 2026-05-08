import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ai" / "models"
data_file = MODELS_DIR / "datos_caracteristicas.pkl"
labels_file = MODELS_DIR / "etiquetas.pkl"
model_file = MODELS_DIR / "modelo_senna.pkl"
encoder_file = MODELS_DIR / "encoder_senna.pkl"

print("📚 ENTRENANDO MODELO DE RECONOCIMIENTO DE SEÑAS")
print("=" * 50)

# Verificar que existan los archivos
if not os.path.exists(data_file) or not os.path.exists(labels_file):
    print("❌ Error: No se encontraron los archivos de datos.")
    print(f"   Verifica que existan:")
    print(f"   - {data_file}")
    print(f"   - {labels_file}")
    print("\n   Primero debes ejecutar: extraer_caracteristicas.py")
    exit()

# Cargar datos
print("\n📂 Cargando datos...")
with open(data_file, 'rb') as f:
    X = pickle.load(f)

with open(labels_file, 'rb') as f:
    y = pickle.load(f)

print(f"✓ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"✓ Etiquetas: {len(np.unique(y))} señas diferentes")

# Codificar etiquetas
print("\n🔤 Codificando etiquetas...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_

print(f"✓ Clases encontradas: {list(classes)}")

# Dividir datos (80% entrenamiento, 20% prueba)
print("\n📊 Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✓ Datos de entrenamiento: {X_train.shape[0]}")
print(f"✓ Datos de prueba: {X_test.shape[0]}")

# Entrenar modelo
print("\n🤖 Entrenando modelo Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    random_state=42,
    n_jobs=1,
    verbose=1
)

model.fit(X_train, y_train)

# Evaluar
print("\n📈 Evaluando modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ PRECISIÓN: {accuracy * 100:.2f}%")
print(f"\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=classes))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de confusión:")
print(cm)

# Guardar modelo
print(f"\n💾 Guardando modelo...")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_file)
joblib.dump(encoder, encoder_file)

print(f"✓ Modelo guardado: {model_file}")
print(f"✓ Encoder guardado: {encoder_file}")

print("\n" + "=" * 50)
print("✓ ¡ENTRENAMIENTO COMPLETADO!")
print("=" * 50)
print(f"\nPróximo paso: Ejecutar 'python usar_modelo.py' para hacer predicciones")
