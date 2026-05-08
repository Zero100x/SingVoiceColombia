# Datasets

Los datasets se mantienen fuera de Git para evitar repositorios pesados y proteger datos de captura.

## Estructura local

```text
datasets/
|- static_signs/     # Imagenes por clase o letra
|- dynamic_signs/    # Secuencias de frames por muestra
|- archives/         # Zips o respaldos locales
```

## Recomendacion

Documentar la procedencia, fecha, cantidad de muestras y condiciones de captura en `docs/dataset.md`.
