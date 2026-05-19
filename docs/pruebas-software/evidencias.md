# Evidencias de Ejecucion de Pruebas

Este archivo organiza las evidencias que deben adjuntarse o enlazarse desde la plataforma colaborativa.

## Evidencias existentes

| Evidencia | Ruta | Casos relacionados |
| --- | --- | --- |
| Plan JMeter | `backend/jmeter/signvoice_load_test.jmx` | TC-CARGA-001 |
| Resultados JMeter | `backend/jmeter/summary.csv` | TC-CARGA-001, TC-LAT-001 |
| Documento JMeter | `docs/PRUEBAS_CARGA_JMETER.md` | TC-DISP-001, TC-LAT-001, TC-CARGA-001 |
| Modelo dinamico | `frontend/android/app/src/main/assets/modelo_dinamico.tflite` | TC-FUN-001 |
| Etiquetas dinamicas | `frontend/android/app/src/main/assets/dynamic_labels.txt` | TC-FUN-001, TC-FUN-004 |
| Metadata del modelo | `ai/models/modelo_dinamico_metadata.json` | TC-REN-001, TC-IA-001 |
| Reporte de clasificacion | `ai/models/reporte_clasificacion_dinamico.txt` | TC-IA-001 |
| Matriz de confusion | `ai/models/matriz_confusion_dinamico.csv` | TC-IA-002 |
| APK debug | `frontend/android/app/build/outputs/apk/debug/app-debug.apk` | TC-COMP-001 |
| Script de prueba IA | `ai/scripts/usar_modelo_dinamico.py` | TC-COMP-002, TC-EXP-002 |
| Captura multicamara | `ai/scripts/capturar_dataset_multicamara.py` | TC-REN-001 |

## Evidencias pendientes recomendadas

Crear una carpeta local:

```text
resources/evidencias-pruebas/
```

Guardar alli:

| Archivo sugerido | Caso | Descripcion |
| --- | --- | --- |
| `TC-EXP-001-sesion-exploratoria.mp4` | TC-EXP-001 | Video usando la app libremente |
| `TC-FUN-002-camara-activa.png` | TC-FUN-002 | Pantallazo de camara activa |
| `TC-FUN-003-reconocimiento-sena.mp4` | TC-FUN-003 | Video de una sena reconocida |
| `TC-ACC-001-contraste-ui.png` | TC-ACC-001 | Captura para accesibilidad |
| `TC-W3C-001-validacion.png` | TC-W3C-001 | Resultado del validador W3C si aplica |
| `TC-CON-001-tracert-github.png` | TC-CON-001 | Pantallazo de `tracert github.com` |
| `TC-SEG-002-manifest.png` | TC-SEG-002 | Captura o evidencia de permisos Android |
| `TC-EXP-002-app-vs-script.mp4` | TC-EXP-002 | Comparacion app Android vs script Python |

## Comandos utiles para evidencias

### Disponibilidad

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health
```

### Conectividad

```powershell
tracert github.com
```

### Script de reconocimiento

```powershell
python ai\scripts\usar_modelo_dinamico.py --camara 1 --debug
```

### Entrenamiento del modelo

```powershell
python ai\scripts\entrenar_modelo_dinamico.py
```

### Instalacion Android

```powershell
cd frontend\android
.\gradlew.bat installDebug
```

## Recomendacion para el tablero

Cada caso de prueba debe tener el enlace o ruta de evidencia en el campo `Evidencia`. Si el caso falla, debe relacionarse con un bug del `registro-defectos.csv` y moverse nuevamente a `To Do`.
