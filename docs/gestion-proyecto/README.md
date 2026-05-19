# Gestion del Proyecto - SignVoice Colombia

Este paquete documenta la gestion colaborativa del proyecto de grado **SignVoice Colombia / Sinbones Colombia**. Esta preparado para montarse en una herramienta como **GitHub Projects**, **Trello**, **Jira** o **Notion**.

## Metodologia usada

El proyecto se gestiona bajo la tecnica de **prototipado evolutivo**. Esto significa que no se construye todo el sistema en una sola version final, sino mediante prototipos incrementales que se prueban, se evaluan y se mejoran.

En este proyecto, cada sprint representa una evolucion del prototipo:

| Prototipo | Enfoque | Resultado |
| --- | --- | --- |
| Prototipo 1 | Estructura base y dataset inicial | Proyecto organizado y primeras capturas |
| Prototipo 2 | Modelo inicial y app Android | Reconocimiento inicial integrado |
| Prototipo 3 | Senas dinamicas y multicamara | Modelo dinamico, clase desconocido y APK funcional |
| Prototipo 4 | Validacion y cierre | Pruebas, tablero de gestion y entrega final |

Cada prototipo incluye implementacion, prueba, hallazgos y ajustes para la siguiente version.

## Herramienta recomendada

Se recomienda usar **GitHub Projects** porque el codigo ya esta en GitHub:

Repositorio: https://github.com/Zero100x/SingVoiceColombia

Estructura sugerida del tablero:

| Columna | Uso |
| --- | --- |
| Product Backlog | Historias y mejoras pendientes |
| Sprint Backlog | Tareas seleccionadas para el sprint actual |
| En progreso | Trabajo que se esta implementando |
| Revision / Pruebas | Tareas terminadas en validacion |
| Hecho | Tareas finalizadas con evidencia |

Campos recomendados:

| Campo | Tipo | Ejemplo |
| --- | --- | --- |
| Prioridad | Select | Alta, Media, Baja |
| Sprint | Select | Sprint 1, Sprint 2, Sprint 3 |
| Tipo | Select | Historia, Tarea, Bug, Prueba, Documentacion |
| Fecha inicio | Date | 2026-05-14 |
| Fecha fin | Date | 2026-05-19 |
| Responsable | Text | Diego |
| Evidencia | Text / Link | Ruta del documento o captura |

## Archivos incluidos

| Archivo | Proposito |
| --- | --- |
| `backlog-producto.csv` | Backlog priorizado para importar o copiar al tablero |
| `tareas-sprints.csv` | Tareas divididas por sprint, fechas, estados y evidencias |
| `sprints.md` | Planeacion de sprints y objetivos |
| `evidencias-pruebas.md` | Evidencias tecnicas de pruebas funcionales, modelo y carga |
| `guion-sustentacion.md` | Texto guia para explicar la gestion del proyecto |
| `prototipado-evolutivo.md` | Trazabilidad de prototipos, hallazgos y mejoras |

## Paquete de pruebas de software

Las pruebas formales del proyecto estan documentadas en:

```text
docs/pruebas-software/
```

Ese paquete incluye plan de pruebas bajo ISO/IEC/IEEE 29119, casos de prueba importables, matriz de trazabilidad, registro de defectos y evidencias. En la herramienta colaborativa se recomienda crear una vista adicional llamada **QA / Pruebas** y cargar el archivo `docs/pruebas-software/casos-prueba.csv`.

## Como montar el proyecto en GitHub Projects

1. Entrar al repositorio en GitHub.
2. Ir a `Projects`.
3. Crear un proyecto nuevo tipo `Board`.
4. Crear las columnas: `Product Backlog`, `Sprint Backlog`, `En progreso`, `Revision / Pruebas`, `Hecho`.
5. Crear los campos personalizados indicados arriba.
6. Copiar los elementos de `backlog-producto.csv` como issues o items del proyecto.
7. Copiar las tareas de `tareas-sprints.csv` y asignarlas al sprint correspondiente.
8. Subir o enlazar evidencias:
   - Documentacion del modelo.
   - Reporte de JMeter.
   - Matriz de confusion.
   - APK generado.
   - Commits del repositorio.
9. Crear una vista **QA / Pruebas** e importar `docs/pruebas-software/casos-prueba.csv`.
10. En `Project settings`, configurar permisos de lectura para el docente.
11. Compartir el enlace del proyecto en la entrega.

## Enlace para la entrega

Cuando el tablero este creado, pegar aqui el enlace:

```text
Enlace del proyecto de gestion: PENDIENTE_DE_PEGAR
```

## Sustentacion breve

El proyecto fue organizado bajo la tecnica de prototipado evolutivo, apoyada por un tablero agil con backlog, sprints, tareas tecnicas, tareas de pruebas y evidencias. La gestion refleja el trabajo real realizado sobre captura de dataset, entrenamiento del modelo de IA, integracion en Android, pruebas con camara, pruebas de carga con JMeter y documentacion tecnica. Cada prototipo genero hallazgos que alimentaron mejoras posteriores, como la clase `desconocido`, la captura multicamara y el ajuste del preprocesamiento de Android.
