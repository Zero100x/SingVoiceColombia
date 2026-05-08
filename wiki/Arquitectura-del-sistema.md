# Arquitectura del Sistema

El sistema se divide en frontend Android, modulo de IA, datasets locales, documentacion y backend futuro. La app consume modelos TensorFlow Lite exportados desde los scripts Python.

```text
Usuario -> App Android -> CameraX -> Modelo TFLite -> Texto/Voz
Dataset local -> Scripts Python -> Modelo IA -> Exportacion Android
```
