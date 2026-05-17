# Pruebas de Carga con JMeter - SignVoice Colombia

## Objetivo

Validar el comportamiento de una API simulada de SignVoice Colombia bajo carga concurrente, usando Apache JMeter con peticiones HTTP `GET` y `POST`.

La app movil actual reconoce señas de forma local con TensorFlow Lite y MediaPipe. Como el backend productivo aun no esta implementado, esta prueba usa un servidor local academico ubicado en `backend/load_test_server.py`. Ese servidor representa endpoints futuros para consultar clases, consultar version del modelo y registrar muestras del dataset.

## Alcance de la prueba

Endpoints evaluados:

- `GET /api/health`: valida disponibilidad del servicio.
- `GET /api/classes`: consulta las clases del alfabeto y conteo de imagenes por clase.
- `GET /api/model-version`: consulta informacion del modelo `.tflite` y labels.
- `POST /api/samples`: registra una muestra simulada del dataset.

## Herramientas

- Apache JMeter.
- Python 3.
- Servidor local `ThreadingHTTPServer`.
- Plan de prueba: `backend/jmeter/signvoice_load_test.jmx`.

## Preparacion

Desde la raiz del proyecto:

```powershell
cd "C:\Users\DIEGO\OneDrive\Documents\Universidad\proyecto de grado\SingVoiceColombia"
.\.venv\Scripts\Activate.ps1
python backend\load_test_server.py --host 127.0.0.1 --port 8080
```

En otra terminal, validar que el servidor responde:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health
Invoke-RestMethod http://127.0.0.1:8080/api/classes
Invoke-RestMethod http://127.0.0.1:8080/api/model-version
```

## Configuracion en JMeter

Abrir JMeter y cargar:

```text
backend/jmeter/signvoice_load_test.jmx
```

Parametros configurados por defecto:

| Parametro | Valor | Descripcion |
| --- | ---: | --- |
| Usuarios concurrentes | 25 | Cantidad de usuarios simulados |
| Ramp-up | 20 s | Tiempo para iniciar gradualmente los usuarios |
| Duracion | 60 s | Tiempo total de la prueba |
| Throughput | 300/min | Limite aproximado de peticiones por minuto |
| Host | 127.0.0.1 | Servidor evaluado |
| Puerto | 8080 | Puerto del servidor |

## Ejecucion por interfaz grafica

1. Abrir JMeter.
2. Cargar `backend/jmeter/signvoice_load_test.jmx`.
3. Revisar el `Thread Group`.
4. Ejecutar con el boton de iniciar.
5. Observar `Summary Report`.

## Ejecucion por consola

Cuando la prueba ya funcione en modo grafico, ejecutarla en modo no grafico:

```powershell
jmeter -n -t backend\jmeter\signvoice_load_test.jmx -l backend\jmeter\resultados_signvoice.jtl -e -o backend\jmeter\reporte_signvoice
```

Ejemplo con usuarios y duracion personalizados:

```powershell
jmeter -n -t backend\jmeter\signvoice_load_test.jmx -l backend\jmeter\resultados_50u.jtl -e -o backend\jmeter\reporte_50u -Jthreads=50 -Jduration=120
```

Para cambiar el limite de peticiones por minuto, abrir en JMeter el temporizador `Limite de peticiones por minuto` y modificar el valor `300.0`.

## Escenarios sugeridos

| Escenario | Usuarios | Duracion | Throughput | Proposito |
| --- | ---: | ---: | ---: | --- |
| Baja carga | 10 | 60 s | 120/min | Validar funcionamiento base |
| Carga media | 25 | 60 s | 300/min | Simular uso normal |
| Carga alta | 50 | 120 s | 600/min | Observar limite del servicio |

## Metricas a reportar

Tomar estos valores desde `Summary Report` o desde el reporte HTML:

- Numero de muestras ejecutadas.
- Tiempo de respuesta promedio.
- Tiempo minimo y maximo.
- Porcentaje de error.
- Throughput o peticiones por segundo.
- Usuarios concurrentes configurados.
- Comportamientos anormales observados.

## Plantilla de resultados

| Escenario | Usuarios | Muestras | Promedio ms | Min ms | Max ms | Error % | Throughput | Observaciones |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baja carga | 10 | | | | | | | |
| Carga media | 25 | | | | | | | |
| Carga alta | 50 | | | | | | | |

## Analisis como tester

Durante la prueba se verifico que los endpoints de consulta y registro respondieran correctamente bajo usuarios concurrentes. Se observaron metricas de tiempo promedio, maximo, errores y rendimiento para determinar si el servicio mantiene estabilidad.

Si el porcentaje de error es `0%` y los tiempos promedio se mantienen bajos, se puede concluir que el servidor soporta la carga simulada. Si aparecen errores HTTP, tiempos maximos muy altos o conexiones interrumpidas, se reportan como hallazgos de rendimiento.

## Sustentacion sugerida

Para la actividad, se puede explicar:

> Se realizo una prueba de carga sobre una API local simulada del proyecto SignVoice Colombia. La prueba fue configurada en Apache JMeter mediante un Thread Group con usuarios concurrentes, ramp-up, duracion y limite de peticiones por minuto. Se probaron endpoints GET para consultar salud del servicio, clases del dataset y version del modelo, ademas de un endpoint POST para registrar muestras del dataset. Durante la ejecucion se monitorearon tiempos de respuesta, numero de usuarios, cantidad de peticiones, errores y throughput. Como tester, el objetivo fue identificar si el servicio presentaba errores, tiempos de respuesta altos, conexiones interrumpidas o caidas bajo diferentes niveles de carga.

## Nota importante

Esta prueba no mide la precision del modelo de inteligencia artificial ni el reconocimiento por camara. JMeter evalua servicios HTTP. La precision del modelo debe evaluarse con matriz de confusion, datos de validacion y pruebas reales de reconocimiento.
