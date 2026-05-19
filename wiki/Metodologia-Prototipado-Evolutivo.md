# Metodologia Prototipado Evolutivo

El proyecto usa **prototipado evolutivo** como tecnica principal de desarrollo. Bajo este enfoque, el sistema no se construye en una sola entrega final, sino mediante prototipos funcionales que se implementan, prueban y refinan.

Esta metodologia es adecuada para SignVoice Colombia porque el reconocimiento de senas depende de experimentacion continua con dataset, camara, modelo de inteligencia artificial, preprocesamiento, interfaz y pruebas en dispositivo real.

## Ciclo aplicado

1. Definir objetivo del prototipo.
2. Implementar una version funcional.
3. Probar con datos reales.
4. Registrar hallazgos.
5. Crear tareas de mejora en el backlog.
6. Evolucionar el sistema en el siguiente prototipo.

## Versiones

| Prototipo | Fechas | Objetivo | Resultado | Hallazgo | Mejora aplicada |
| --- | --- | --- | --- | --- | --- |
| v0.1.0 | 2026-05-07 a 2026-05-10 | Estructura base y dataset inicial | Proyecto organizado y primeras capturas | Dataset necesitaba orden por clases | Documentacion y estructura de carpetas |
| v0.2.0 | 2026-05-10 a 2026-05-14 | Modelo inicial y app Android | Modelo TFLite integrado | Predicciones inestables | Umbrales, confianza y votacion |
| v0.3.0 | 2026-05-15 a 2026-05-18 | Senas dinamicas y multicamara | Modelo dinamico y APK funcional | El sistema inventaba senas | Clase `desconocido` y captura multicamara |
| v0.4.0 | 2026-05-18 a 2026-05-19 | Alineacion app vs script Python | Android usa luminancia YUV | Diferencia entre Python y app | Ajuste de preprocesamiento y ventana temporal |
| v0.5.0 | 2026-05-19 a 2026-05-25 | Validacion y cierre | Gestion, pruebas y evidencias | Faltaba formalizacion en plataforma | Backlog, QA y trazabilidad |

## Relacion con el tablero

Cada prototipo se representa como un sprint o iteracion en la herramienta colaborativa. Los hallazgos de pruebas se transforman en tareas, bugs o mejoras del backlog.

Ejemplos:

- Predicciones aleatorias -> tarea para agregar clase `desconocido`.
- App diferente al script -> bug para alinear preprocesamiento Android.
- Senas con bajo desempeno -> tarea para grabar mas muestras y reentrenar.
- Falta de evidencias -> tarea QA para adjuntar capturas, videos y reportes.
