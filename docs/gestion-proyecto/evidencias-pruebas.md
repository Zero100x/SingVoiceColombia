# Evidencias de Pruebas - SignVoice Colombia

Este documento resume las pruebas realizadas y las evidencias disponibles para la actividad final de Metodologia de Software Colaborativo.

## 1. Pruebas de carga con JMeter

**Objetivo:** validar endpoints HTTP simulados asociados al proyecto.

**Archivo principal:** `backend/jmeter/signvoice_load_test.jmx`

**Endpoints evaluados:**

| Metodo | Endpoint | Proposito |
| --- | --- | --- |
| GET | `/api/health` | Validar disponibilidad del servidor |
| GET | `/api/classes` | Consultar clases del dataset |
| GET | `/api/model-version` | Consultar version del modelo |
| POST | `/api/samples` | Registrar muestra simulada |

**Configuracion usada en prueba de carga alta:**

| Parametro | Valor |
| --- | ---: |
| Usuarios concurrentes | 50 |
| Ramp-up | 120 s |
| Duracion | 120 s |
| Herramienta | Apache JMeter 5.6.3 |

**Resultado observado en Summary Report:**

| Metrica | Valor |
| --- | ---: |
| Total de muestras | 1552 |
| Tiempo promedio total | 554 ms |
| Tiempo minimo | 1 ms |
| Tiempo maximo | 4619 ms |
| Porcentaje de error total | 0.13% |
| Throughput total | 38.9/min |

**Analisis como tester:**

La prueba permitio validar que los endpoints simulados respondieran bajo concurrencia. El porcentaje de error fue bajo, por lo que el servicio academico se mantuvo estable. El endpoint con mayor tiempo promedio fue `GET /api/classes`, debido a que consulta informacion asociada al dataset. No se evidencio caida general del servicio.

**Evidencias:**

- `docs/PRUEBAS_CARGA_JMETER.md`
- `backend/jmeter/signvoice_load_test.jmx`
- `backend/jmeter/summary.csv`

## 2. Pruebas del modelo de inteligencia artificial

**Modelo evaluado:** modelo dinamico TensorFlow Lite.

**Archivo del modelo:** `frontend/android/app/src/main/assets/modelo_dinamico.tflite`

**Archivo de etiquetas:** `frontend/android/app/src/main/assets/dynamic_labels.txt`

**Clases entrenadas:**

- ABRAZAR
- ABURRIDO
- ACEPTAR
- ACOMPANAR
- AHORA
- AMIGO
- AQUI
- ARRIBA
- ATENCION
- BUENOS_DIAS
- CON_GUSTO
- DISCULPA
- FAMILIA
- GRACIAS
- HASTA_MANANA
- HOLA
- HOLA2
- MAMA
- DESCONOCIDO

**Metricas del ultimo entrenamiento:**

| Metrica | Valor |
| --- | ---: |
| Precision de prueba | 86.61% |
| Perdida de prueba | 0.8226 |
| Entrada del modelo | 96x96x6 |
| Longitud temporal | 16 frames |
| Clase desconocido | Activada |
| Ventanas omitidas por bajo movimiento | 1997 |

**Hallazgos:**

- El modelo reconoce varias senas correctamente, pero algunas clases siguen siendo debiles.
- Las clases con 1 muestra no se entrenaron como senas reales y fueron usadas para reforzar `desconocido`.
- La clase `DESCONOCIDO` mejora la seguridad del sistema porque evita predicciones inventadas cuando no hay una sena clara.
- La app Android requirio ajuste de preprocesamiento para usar luminancia YUV, mas cercana al script Python.

**Evidencias:**

- `ai/models/reporte_clasificacion_dinamico.txt`
- `ai/models/matriz_confusion_dinamico.csv`
- `ai/models/modelo_dinamico_metadata.json`

## 3. Pruebas funcionales en app movil

**Objetivo:** validar reconocimiento en dispositivo Android real.

**Dispositivo usado:** TECNO KG5k - Android 11.

**APK:** `frontend/android/app/build/outputs/apk/debug/app-debug.apk`

**Validaciones realizadas:**

| Prueba | Resultado esperado | Estado |
| --- | --- | --- |
| Abrir app instalada | App inicia sin errores | Realizada |
| Activar camara | Vista de camara visible | Realizada |
| Cargar modelo dinamico | No debe mostrar modelo pendiente | Realizada |
| Detectar sena dinamica | Muestra resultado si hay confianza | En ajuste |
| Evitar falsas predicciones | Usa desconocido / baja confianza | En ajuste |
| Audio automatico | Lee traduccion si esta activado | Realizada |

## 4. Pruebas de scripts Python

**Scripts revisados:**

| Script | Proposito |
| --- | --- |
| `ai/scripts/usar_modelo_dinamico.py` | Probar modelo dinamico con camara en PC |
| `ai/scripts/entrenar_modelo_dinamico.py` | Entrenar y exportar modelo dinamico |
| `ai/scripts/capturar_dataset_multicamara.py` | Capturar muestras desde 3 camaras |

**Comandos utiles:**

```powershell
python ai\scripts\usar_modelo_dinamico.py --camara 1 --debug
python ai\scripts\usar_modelo_dinamico.py --camara 1 --debug --fps-inferencia 5
python ai\scripts\entrenar_modelo_dinamico.py
```

## 5. Conclusiones de calidad

El proyecto cuenta con pruebas de carga, pruebas de entrenamiento, pruebas funcionales en dispositivo real y pruebas manuales de reconocimiento. El sistema aun requiere mejora continua en precision, especialmente aumentando muestras de las clases mas debiles y agregando diagnostico visual en Android para comparar predicciones contra el script Python.
