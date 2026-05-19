# Prototipado Evolutivo Aplicado a SignVoice Colombia

## Enfoque metodologico

El proyecto utiliza **prototipado evolutivo** como tecnica principal de desarrollo. Bajo este enfoque, el sistema se construye por versiones funcionales sucesivas. Cada prototipo se implementa, se prueba con datos reales, se documentan sus problemas y luego se mejora en la siguiente iteracion.

Esta tecnica es adecuada para SignVoice Colombia porque el reconocimiento de senas depende de experimentacion continua con dataset, camara, modelo de inteligencia artificial, interfaz y pruebas en dispositivo real.

## Ciclo de trabajo

Cada iteracion sigue este ciclo:

1. Definir objetivo del prototipo.
2. Implementar una version funcional.
3. Probar con dataset, camara o usuario.
4. Registrar hallazgos.
5. Crear tareas de mejora en el backlog.
6. Evolucionar el sistema en el siguiente prototipo.

## Trazabilidad de prototipos

| Prototipo | Fechas | Objetivo | Resultado | Hallazgo principal | Mejora aplicada |
| --- | --- | --- | --- | --- | --- |
| Prototipo 1 | 2026-05-07 a 2026-05-10 | Crear estructura inicial y dataset base | Repositorio, guias y primeras capturas | El dataset necesitaba orden por clases | Se documento estructura y flujo de datos |
| Prototipo 2 | 2026-05-10 a 2026-05-14 | Integrar modelo inicial en app Android | Modelo TFLite y APK inicial | La app mostraba predicciones inestables | Se agregaron umbrales, confianza y votacion temporal |
| Prototipo 3 | 2026-05-15 a 2026-05-18 | Reconocer senas dinamicas | Modelo dinamico con 19 clases | El sistema inventaba senas cuando no habia gesto claro | Se agrego clase `desconocido` y captura multicamara |
| Prototipo 4 | 2026-05-18 a 2026-05-19 | Alinear app con script Python | Android usa luminancia YUV y ventana temporal ajustada | La app no funcionaba igual que el script | Se ajusto preprocesamiento y tiempos de inferencia |
| Prototipo 5 | 2026-05-19 a 2026-05-25 | Validacion final y gestion del proyecto | Backlog, sprints, pruebas y evidencias | Faltaba formalizar gestion en herramienta | Se preparo paquete para GitHub Projects/Trello/Jira |

## Relacion con el backlog

El backlog no es estatico. Se refina a partir de los resultados de cada prototipo. Por ejemplo:

- Al detectar predicciones aleatorias, se creo la tarea de agregar umbrales y clase `desconocido`.
- Al ver que el script Python funcionaba mejor que la app, se creo la tarea de alinear el preprocesamiento Android.
- Al notar que algunas senas dinamicas requieren mas variacion, se creo la tarea de captura multicamara.
- Al identificar clases debiles, se dejo pendiente grabar mas muestras y reentrenar.

## Evidencias por prototipo

| Prototipo | Evidencias |
| --- | --- |
| Prototipo 1 | `docs/GUIA_DE_INICIO.md`, `docs/dataset.md` |
| Prototipo 2 | `frontend/android/app/src/main/assets/`, `docs/modelo-ia.md` |
| Prototipo 3 | `ai/scripts/capturar_dataset_multicamara.py`, `ai/models/reporte_clasificacion_dinamico.txt` |
| Prototipo 4 | `DynamicRecognitionService.kt`, `MainActivity.kt`, APK instalado |
| Prototipo 5 | `docs/gestion-proyecto/`, tablero de gestion |

## Como explicarlo en la sustentacion

> Nuestra metodologia fue prototipado evolutivo. No construimos una version final desde el inicio; fuimos creando prototipos funcionales. Primero organizamos el dataset, luego entrenamos un modelo inicial, despues lo integramos en Android, luego agregamos senas dinamicas, captura multicamara y clase desconocido. Cada prueba genero hallazgos y esos hallazgos se convirtieron en nuevas tareas del backlog. Por eso el tablero refleja una evolucion real del proyecto.
