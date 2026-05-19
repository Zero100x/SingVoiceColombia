# Plan de Pruebas - ISO/IEC/IEEE 29119

## 1. Identificacion

**Proyecto:** SignVoice Colombia / Sinbones Colombia  
**Tipo de sistema:** Aplicacion movil con IA, scripts de entrenamiento y API local academica  
**Tecnica de desarrollo:** Prototipado evolutivo  
**Responsable:** Diego  
**Fecha:** 2026-05-19

## 2. Objetivo del plan

Definir el proceso de pruebas para verificar que el sistema cumpla con sus requisitos funcionales y no funcionales, incluyendo reconocimiento de senas, carga del modelo, estabilidad de la app, rendimiento de endpoints, seguridad basica, accesibilidad y compatibilidad.

## 3. Referencia a ISO/IEC/IEEE 29119

El plan aplica lineamientos generales de la norma:

- Planificacion de pruebas.
- Diseno de casos de prueba.
- Registro de condiciones de prueba.
- Ejecucion controlada.
- Evidencia del resultado.
- Gestion de incidentes y defectos.
- Revalidacion despues de correcciones.

## 4. Elementos bajo prueba

| Elemento | Descripcion |
| --- | --- |
| App Android | Pantallas, camara, reconocimiento, audio y permisos |
| Modelo IA | Modelo dinamico `.tflite`, etiquetas y precision |
| Scripts Python | Captura multicamara, entrenamiento y uso del modelo |
| API local | Endpoints academicos para pruebas HTTP |
| Documentacion | Guias, gestion del proyecto y evidencias |

## 5. Tipos de prueba

| Tipo | Aplicacion en el proyecto |
| --- | --- |
| Exploratoria | Uso libre de app y scripts para descubrir fallos |
| Funcional | Validar camara, modelo, audio, permisos y traduccion |
| Usabilidad | Validar claridad de interfaz y mensajes |
| Accesibilidad | Contraste, tamanos de texto, botones y descripcion |
| W3C | Validacion de recursos web/reporte HTML cuando aplique |
| Disponibilidad | Verificar API local y app en ejecucion |
| Latencia | Medir tiempos de respuesta de API y reconocimiento |
| Conectividad/tracert | Validar ruta de red hacia servicios o repositorio |
| Rendimiento | Medir comportamiento con procesamiento y modelo |
| Carga | Simular usuarios concurrentes con JMeter |
| Compatibilidad | Probar en celular real y ambiente Windows |
| Seguridad basica | Revisar permisos, archivos sensibles y endpoints |
| IA/calidad del modelo | Matriz de confusion, accuracy y clase desconocido |

## 6. Ambiente de pruebas

| Recurso | Detalle |
| --- | --- |
| Sistema operativo | Windows |
| Movil | TECNO KG5k - Android 11 |
| Python | Entorno virtual `.venv` |
| Android | Gradle / APK debug |
| JMeter | Apache JMeter 5.6.3 |
| Modelo | TensorFlow / TensorFlow Lite |
| Repositorio | GitHub |

## 7. Criterios de entrada

- Repositorio disponible.
- Entorno virtual creado.
- Dataset organizado por clases.
- Modelo `.tflite` disponible en assets.
- App compila en modo debug.
- Plan JMeter disponible.
- Casos de prueba registrados en la plataforma colaborativa.

## 8. Criterios de salida

- Casos de prueba ejecutados o justificados.
- Defectos registrados.
- Evidencias asociadas.
- Pruebas fallidas retornadas al flujo `To Do` o equivalente.
- Reportes actualizados.
- Enlace del tablero compartido.

## 9. Riesgos

| Riesgo | Impacto | Mitigacion |
| --- | --- | --- |
| Dataset desbalanceado | Baja precision | Grabar mas muestras por clase |
| Diferencia entre script y app | Resultados inconsistentes | Alinear preprocesamiento |
| Camara o iluminacion variable | Falsas predicciones | Usar clase desconocido y recomendaciones de captura |
| Backend productivo no implementado | Pruebas web limitadas | Usar API local academica |
| Falta de evidencia visual | Baja trazabilidad | Guardar pantallazos o videos por prueba |

## 10. Gestion de defectos

Cuando un caso falle:

1. Se registra en `registro-defectos.csv`.
2. Se crea item en el tablero.
3. Se relaciona con el caso de prueba.
4. Se mueve a `To Do`.
5. Se corrige.
6. Se reejecuta el caso.
7. Se actualiza el estado a `Aprobada` o `Fallida`.
