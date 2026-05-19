# Pruebas de Software - SignVoice Colombia

Este paquete documenta las pruebas de software del proyecto de grado **SignVoice Colombia / Sinbones Colombia**, aplicando buenas practicas alineadas con la norma **ISO/IEC/IEEE 29119** para procesos de pruebas de software.

## Objetivo

Registrar, gestionar y evidenciar pruebas que permitan validar la calidad, estabilidad, seguridad, disponibilidad, rendimiento y correcto funcionamiento de la solucion desarrollada.

## Alcance

El proyecto incluye:

- Aplicacion Android para reconocimiento de senas.
- Scripts Python para captura, entrenamiento y prueba de modelos.
- Modelo de inteligencia artificial exportado a TensorFlow Lite.
- API local academica para pruebas de carga con JMeter.
- Documentacion tecnica y gestion del proyecto.

## Estructura del paquete

| Archivo | Proposito |
| --- | --- |
| `plan-pruebas-iso29119.md` | Estrategia, alcance, ambiente, criterios de entrada/salida y flujo de defectos |
| `casos-prueba.csv` | Casos de prueba listos para cargar en GitHub Projects, Jira, GitLab o Trello |
| `matriz-trazabilidad.csv` | Relacion entre requisitos, casos de prueba y evidencias |
| `registro-defectos.csv` | Registro de fallos y retorno al flujo de trabajo |
| `evidencias.md` | Lista organizada de evidencias de ejecucion |
| `guion-sustentacion-pruebas.md` | Texto guia para explicar las pruebas en clase |

## Flujo de gestion recomendado

En la plataforma colaborativa, crear una vista llamada **QA / Pruebas** con columnas:

| Columna | Uso |
| --- | --- |
| To Do | Pruebas pendientes o fallidas que requieren correccion |
| En Proceso | Pruebas en ejecucion |
| En Revision | Pruebas ejecutadas esperando validacion |
| Aprobada | Pruebas exitosas con evidencia |
| Fallida | Pruebas con defecto reportado |
| Revalidacion | Pruebas que se repiten despues de corregir un bug |

Campos minimos por caso:

- Identificador.
- Nombre.
- Objetivo.
- Precondiciones.
- Datos de entrada.
- Pasos de ejecucion.
- Resultado esperado.
- Resultado obtenido.
- Estado.
- Evidencia.
- Responsable.
- Tipo de prueba.

## Estados permitidos

| Estado | Descripcion |
| --- | --- |
| Pendiente | Caso creado, aun no ejecutado |
| En Proceso | Caso en ejecucion |
| Aprobada | Resultado obtenido coincide con esperado |
| Fallida | Resultado obtenido no coincide con esperado |
| Bloqueada | No se puede ejecutar por dependencia externa |
| Revalidacion | Se repite despues de corregir un defecto |

## Regla para pruebas fallidas

Si una prueba falla:

1. Registrar el defecto en `registro-defectos.csv`.
2. Crear una tarea o issue asociada en el tablero.
3. Mover la tarea al estado `To Do` o `Inicio`.
4. Corregir el bug.
5. Repetir la prueba.
6. Actualizar evidencia y estado final.

## Enlace a la plataforma colaborativa

Cuando el tablero este creado, pegar aqui el enlace:

```text
Enlace QA / Pruebas: PENDIENTE_DE_PEGAR
```
