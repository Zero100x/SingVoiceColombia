# Guion de Sustentacion - Gestion del Proyecto

## Introduccion

Para la actividad final de Metodologia de Software Colaborativo se implemento la gestion del proyecto de grado **SignVoice Colombia**, una aplicacion orientada al reconocimiento de Lengua de Senas Colombiana mediante camara, inteligencia artificial y traduccion a texto/voz.

El proyecto fue organizado en una herramienta de gestion usando un tablero agil con backlog, tareas, sprints, fechas, estados de avance y evidencias de pruebas. La metodologia aplicada fue **prototipado evolutivo**.

## 0. Metodologia: prototipado evolutivo

La tecnica utilizada fue prototipado evolutivo. Esto significa que el sistema no se construyo de una sola vez, sino mediante versiones funcionales que fueron mejorando con base en pruebas reales.

En SignVoice Colombia se trabajo asi:

- Primer prototipo: estructura del proyecto y dataset inicial.
- Segundo prototipo: modelo inicial integrado en Android.
- Tercer prototipo: reconocimiento dinamico con captura multicamara.
- Cuarto prototipo: mejoras de precision, clase desconocido y ajuste del preprocesamiento de Android.
- Quinto prototipo: validacion, documentacion, tablero y entrega final.

Cada prototipo genero hallazgos. Por ejemplo, cuando el sistema lanzaba senas al azar, se agrego la clase `desconocido`; cuando el script Python funcionaba mejor que la app, se ajusto el preprocesamiento de Android para usar luminancia YUV.

## 1. Herramienta de gestion

Se selecciono una herramienta tipo tablero agil, recomendada como **GitHub Projects**, porque el proyecto ya se encuentra versionado en GitHub. El tablero se adapto al prototipado evolutivo usando sprints como iteraciones del prototipo.

El tablero se organizo con las columnas:

- Product Backlog.
- Sprint Backlog.
- En progreso.
- Revision / Pruebas.
- Hecho.

Ademas, se definieron campos como prioridad, sprint, fecha de inicio, fecha de finalizacion, responsable, tipo de tarea y evidencia.

## 2. Backlog del producto

El backlog incluye historias de usuario, tareas tecnicas, bugs, pruebas y documentacion. Cada elemento esta priorizado y relacionado con los objetivos del proyecto.

Ejemplos:

- Capturar dataset de senas.
- Entrenar modelo de inteligencia artificial.
- Integrar modelo TensorFlow Lite en Android.
- Evitar predicciones aleatorias usando confianza, margen y clase desconocido.
- Realizar pruebas de carga con JMeter.
- Documentar evidencias del proyecto.

## 3. Sprints

El trabajo fue dividido en cuatro sprints, cada uno relacionado con una evolucion del prototipo:

| Sprint | Fechas | Objetivo |
| --- | --- | --- |
| Sprint 1 | 2026-05-07 a 2026-05-10 | Prototipo base: fundamentos del proyecto y dataset inicial |
| Sprint 2 | 2026-05-10 a 2026-05-14 | Prototipo funcional: modelo inicial, app movil y pruebas de carga |
| Sprint 3 | 2026-05-15 a 2026-05-18 | Prototipo dinamico: multicamara, clase desconocido y precision |
| Sprint 4 | 2026-05-19 a 2026-05-25 | Prototipo final evaluable: cierre, tablero y sustentacion |

Cada sprint tiene tareas con fecha, prioridad, estado y evidencia asociada.

## 4. Seguimiento del trabajo

El seguimiento se realiza moviendo las tareas entre columnas del tablero. Por ejemplo, una tarea inicia en backlog, pasa a sprint backlog, luego a en progreso, despues a revision/pruebas y finalmente a hecho.

Esto permite evidenciar el avance real del proyecto y controlar las actividades pendientes.

## 5. Evidencias de pruebas

Se documentaron tres tipos principales de pruebas:

1. **Pruebas de carga con JMeter**
   - Se probaron endpoints GET y POST.
   - Se usaron 50 usuarios concurrentes.
   - Se obtuvo 0.13% de error total.

2. **Pruebas del modelo de IA**
   - El modelo dinamico alcanzo 86.61% de precision de prueba.
   - Se genero matriz de confusion y reporte de clasificacion.
   - Se agrego la clase `DESCONOCIDO` para reducir falsas predicciones.

3. **Pruebas funcionales en Android**
   - Se compilo e instalo APK en dispositivo real.
   - Se valido carga del modelo, camara, reconocimiento y salida de voz.

## 6. Analisis como equipo

La gestion del proyecto permitio identificar problemas reales durante el desarrollo, como baja precision, diferencias entre script y app, necesidad de capturar mas muestras y ajustes en el preprocesamiento de la camara.

El proyecto evidencia mejora continua porque las tareas no solo se marcaron como hechas, sino que generaron nuevas tareas de correccion y refinamiento, tal como lo propone el prototipado evolutivo.

## 7. Cierre

Como resultado, el proyecto cuenta con backlog, tareas, sprints, fechas, seguimiento y evidencias. El tablero facilita la revision del docente y muestra un proceso de trabajo realista, iterativo y alineado con prototipado evolutivo y buenas practicas de gestion de proyectos.

## Frase final sugerida

> En conclusion, SignVoice Colombia no se gestiono solo como una entrega de codigo, sino como un proyecto en evolucion, con planeacion, pruebas, control de avance y mejora continua basada en resultados reales.
