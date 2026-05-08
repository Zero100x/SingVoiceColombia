# Manual Tecnico

## Entrenar modelo clasico

```powershell
python ai\scripts\extraer_caracteristicas.py
python ai\scripts\entrenar_modelo.py
```

## Entrenar modelo TensorFlow Lite

```powershell
python ai\scripts\entrenar_modelo_tflite.py
```

## Compilar Android

```powershell
cd frontend\android
.\gradlew.bat assembleDebug
```
