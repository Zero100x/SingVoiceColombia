# Planeacion de Sprints - SignVoice Colombia

## Vision del producto

SignVoice Colombia es una aplicacion movil y un conjunto de scripts de apoyo para reconocer senas de Lengua de Senas Colombiana mediante camara, modelos de inteligencia artificial y traduccion a texto/voz.

## Objetivo general de gestion

Organizar el proyecto de grado como un entorno real de trabajo colaborativo, con backlog priorizado, tareas por sprint, fechas, seguimiento y evidencias de pruebas.

## Relacion con prototipado evolutivo

Los sprints representan iteraciones de **prototipado evolutivo**. Cada sprint produce un prototipo funcional, se valida con pruebas y genera aprendizajes para el siguiente ciclo. Por eso el backlog contiene tanto funcionalidades nuevas como bugs, ajustes de precision, pruebas y documentacion.

| Sprint | Prototipo evolutivo | Aprendizaje generado |
| --- | --- | --- |
| Sprint 1 | Prototipo base | Se necesitaba organizar dataset y flujo tecnico |
| Sprint 2 | Prototipo con modelo inicial | El reconocimiento requeria control de confianza y pruebas |
| Sprint 3 | Prototipo dinamico | El sistema necesitaba clase desconocido, multicamara y ajuste Android |
| Sprint 4 | Prototipo final evaluable | Se formaliza gestion, validacion y entrega |

## Sprint 1: Fundamentos del Proyecto

**Fechas:** 2026-05-07 a 2026-05-10  
**Objetivo:** Crear la base tecnica del proyecto, organizar carpetas y preparar la guia inicial.

**Entregables:**

- Repositorio organizado.
- Documentacion base.
- Dataset inicial estructurado.
- Primeras pruebas de camara y deteccion.

**Estado:** Finalizado.

## Sprint 2: Modelo Inicial, App y Pruebas de Carga

**Fechas:** 2026-05-10 a 2026-05-14  
**Objetivo:** Entrenar un primer modelo, integrarlo en Android y documentar pruebas de carga con JMeter.

**Entregables:**

- Modelo exportado a TensorFlow Lite.
- App Android con carga de modelo desde assets.
- Plan de pruebas JMeter.
- Documento de pruebas de carga.

**Evidencias:**

- `frontend/android/app/src/main/assets/modelo_dinamico.tflite`
- `backend/jmeter/signvoice_load_test.jmx`
- `docs/PRUEBAS_CARGA_JMETER.md`

**Estado:** Finalizado.

## Sprint 3: Reconocimiento Dinamico y Mejoras de Precision

**Fechas:** 2026-05-15 a 2026-05-18  
**Objetivo:** Capturar senas dinamicas, mejorar el entrenamiento, agregar clase desconocido y alinear app con script Python.

**Entregables:**

- Captura multicamara.
- Modelo dinamico con 19 clases.
- Clase `DESCONOCIDO`.
- APK instalado en dispositivo real.
- Interfaz de traduccion mejorada.
- Audio automatico activable/desactivable.

**Metricas:**

- Precision de prueba del modelo dinamico: 86.61%.
- Clase `desconocido` con F1 aproximado de 0.83.
- Dataset dinamico con multiples clases y muestras por camara.

**Estado:** Finalizado.

## Sprint 4: Cierre, Validacion y Entrega

**Fechas:** 2026-05-19 a 2026-05-25  
**Objetivo:** Preparar la entrega final, montar la herramienta de gestion, completar evidencias y validar el APK.

**Entregables esperados:**

- Tablero en herramienta de gestion.
- Backlog priorizado.
- Tareas con fechas, responsables y estados.
- Evidencias de pruebas funcionales, IA y carga.
- Enlace compartido para revision.
- Guion de sustentacion.

**Estado:** En progreso.

## Seguimiento sugerido

| Fecha | Actividad | Resultado esperado |
| --- | --- | --- |
| 2026-05-19 | Crear tablero de gestion | Proyecto visible para revision |
| 2026-05-20 | Probar app con senas principales | Lista de aciertos y fallos |
| 2026-05-21 | Grabar muestras faltantes | Dataset reforzado |
| 2026-05-22 | Reentrenar si aplica | Modelo actualizado |
| 2026-05-23 | Actualizar evidencias | Documentacion final |
| 2026-05-24 | Preparar sustentacion | Guion listo |
| 2026-05-25 | Compartir enlace | Entrega final |

## Definicion de terminado

Una tarea se considera terminada cuando:

- Esta implementada o documentada.
- Tiene evidencia asociada.
- Fue probada si afecta funcionalidad.
- El estado del tablero fue actualizado.
- No rompe el flujo principal de la app.
