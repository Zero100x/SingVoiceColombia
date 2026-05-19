# Casos de Prueba

Los casos de prueba estan registrados en:

```text
docs/pruebas-software/casos-prueba.csv
```

Cada caso contiene:

- Identificador.
- Nombre.
- Tipo.
- Objetivo.
- Precondiciones.
- Datos de entrada.
- Pasos de ejecucion.
- Resultado esperado.
- Resultado obtenido.
- Estado.
- Evidencia.
- Responsable.
- Retorno al flujo en caso de falla.

## Casos principales

| ID | Tipo | Nombre | Estado |
| --- | --- | --- | --- |
| TC-EXP-001 | Exploratoria | Exploracion general de app movil | Pendiente |
| TC-FUN-001 | Funcional | Carga de modelo dinamico | Aprobada |
| TC-FUN-002 | Funcional | Activacion de camara | Aprobada |
| TC-FUN-003 | Funcional | Reconocimiento de sena dinamica | En Proceso |
| TC-FUN-004 | Funcional | Clase desconocido | Aprobada |
| TC-FUN-005 | Funcional | Audio automatico | Aprobada |
| TC-USAB-001 | Usabilidad | Claridad de interfaz de camara | En Proceso |
| TC-ACC-001 | Accesibilidad | Contraste y lectura de interfaz | Pendiente |
| TC-ACC-002 | Accesibilidad | Descripcion de botones principales | Aprobada |
| TC-W3C-001 | W3C | Validacion W3C de recursos web | Pendiente |
| TC-DISP-001 | Disponibilidad | Disponibilidad API health | Aprobada |
| TC-LAT-001 | Latencia | Latencia endpoints API | Aprobada |
| TC-CON-001 | Conectividad | Comando tracert al repositorio | Pendiente |
| TC-REN-001 | Rendimiento | Rendimiento entrenamiento modelo | Aprobada |
| TC-CARGA-001 | Carga | Prueba de carga JMeter 50 usuarios | Aprobada |
| TC-COMP-001 | Compatibilidad | Compatibilidad en dispositivo Android | Aprobada |
| TC-COMP-002 | Compatibilidad | Compatibilidad script Python en Windows | Aprobada |
| TC-SEG-001 | Seguridad | Revision de secretos y archivos sensibles | Aprobada |
| TC-SEG-002 | Seguridad | Permisos de camara | Pendiente |
| TC-IA-001 | IA | Reporte de clasificacion modelo dinamico | Aprobada |
| TC-IA-002 | IA | Matriz de confusion modelo dinamico | Aprobada |
| TC-EXP-002 | Exploratoria | Exploracion app vs script | En Proceso |
| TC-USAB-002 | Usabilidad | Tiempo de lectura del resultado | Aprobada |

## Gestion en plataforma colaborativa

Se recomienda importar estos casos en una vista `QA / Pruebas`, con columnas:

- To Do.
- En Proceso.
- En Revision.
- Aprobada.
- Fallida.
- Revalidacion.
