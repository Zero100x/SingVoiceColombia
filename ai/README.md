# Inteligencia Artificial

Este modulo contiene los scripts y salidas tecnicas relacionadas con captura, procesamiento, entrenamiento e inferencia del reconocimiento de sennas.

## Estructura

```text
ai/
|- scripts/
|  |- capturar_dataset.py
|  |- extraer_caracteristicas.py
|  |- entrenar_modelo.py
|  |- entrenar_modelo_tflite.py
|  |- entrenar_modelo_dinamico.py
|  |- usar_modelo.py
|  |- usar_modelo_tflite.py
|  |- usar_modelo_dinamico.py
|- models/
```

## Notas

- Los modelos pesados se generan localmente y no se versionan.
- Las metricas ligeras, reportes y metadata se pueden versionar cuando aporten evidencia academica.
- Los scripts resuelven rutas desde la raiz del proyecto para funcionar en Windows.
