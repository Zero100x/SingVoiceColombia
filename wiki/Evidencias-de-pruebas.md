# Evidencias de Pruebas

Las evidencias permiten demostrar la ejecucion real de las pruebas.

## Evidencias existentes

| Evidencia | Ruta | Casos relacionados |
| --- | --- | --- |
| Plan JMeter | `backend/jmeter/signvoice_load_test.jmx` | TC-CARGA-001 |
| Resultados JMeter | `backend/jmeter/summary.csv` | TC-CARGA-001, TC-LAT-001 |
| Documento JMeter | `docs/PRUEBAS_CARGA_JMETER.md` | TC-DISP-001, TC-LAT-001 |
| Modelo dinamico | `frontend/android/app/src/main/assets/modelo_dinamico.tflite` | TC-FUN-001 |
| Etiquetas dinamicas | `frontend/android/app/src/main/assets/dynamic_labels.txt` | TC-FUN-001, TC-FUN-004 |
| Metadata IA | `ai/models/modelo_dinamico_metadata.json` | TC-REN-001, TC-IA-001 |
| Reporte de clasificacion | `ai/models/reporte_clasificacion_dinamico.txt` | TC-IA-001 |
| Matriz de confusion | `ai/models/matriz_confusion_dinamico.csv` | TC-IA-002 |
| APK debug | `frontend/android/app/build/outputs/apk/debug/app-debug.apk` | TC-COMP-001 |
| Script IA | `ai/scripts/usar_modelo_dinamico.py` | TC-COMP-002 |

## Evidencias pendientes

Guardar en:

```text
resources/evidencias-pruebas/
```

Sugeridas:

| Archivo | Caso | Descripcion |
| --- | --- | --- |
| `TC-EXP-001-sesion-exploratoria.mp4` | TC-EXP-001 | Video usando la app |
| `TC-FUN-002-camara-activa.png` | TC-FUN-002 | Pantallazo de camara activa |
| `TC-FUN-003-reconocimiento-sena.mp4` | TC-FUN-003 | Video de una sena reconocida |
| `TC-ACC-001-contraste-ui.png` | TC-ACC-001 | Evidencia de accesibilidad |
| `TC-W3C-001-validacion.png` | TC-W3C-001 | Resultado validador W3C |
| `TC-CON-001-tracert-github.png` | TC-CON-001 | Evidencia de `tracert github.com` |
| `TC-EXP-002-app-vs-script.mp4` | TC-EXP-002 | Comparacion app vs script |

## Resultados destacados

- JMeter con 50 usuarios concurrentes: 1552 muestras, 554 ms promedio, 0.13% error.
- Modelo dinamico: 86.61% accuracy de prueba.
- Clase `desconocido`: usada para reducir falsas predicciones.
- APK instalado en dispositivo Android real.
