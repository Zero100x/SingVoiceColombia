# Modelo de IA

## Enfoques disponibles

El proyecto incluye dos lineas de reconocimiento:

- Modelo clasico con caracteristicas HOG, histogramas de color y Random Forest.
- Modelo CNN exportable a TensorFlow Lite para integracion movil.

## Modelo clasico

Scripts principales:

- `ai/scripts/extraer_caracteristicas.py`
- `ai/scripts/entrenar_modelo.py`
- `ai/scripts/usar_modelo.py`

Este enfoque es util para validar rapidamente el dataset y experimentar con caracteristicas interpretables.

## Modelo TensorFlow Lite

Scripts principales:

- `ai/scripts/entrenar_modelo_tflite.py`
- `ai/scripts/usar_modelo_tflite.py`

El modelo procesa una region central de la imagen, genera canales de luminancia y bordes, entrena una CNN ligera y exporta un archivo `.tflite` para Android.

## Modelo dinamico

Scripts principales:

- `ai/scripts/entrenar_modelo_dinamico.py`
- `ai/scripts/usar_modelo_dinamico.py`

Este enfoque usa secuencias de frames para sennas con movimiento. Forma parte de las iteraciones futuras del prototipo.

## Artefactos

Los artefactos pesados quedan en `ai/models` y no se suben a Git. Los reportes ligeros pueden versionarse como evidencia academica.
