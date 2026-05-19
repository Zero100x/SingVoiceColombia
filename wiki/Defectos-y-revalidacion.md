# Defectos y Revalidacion

El proyecto registra defectos cuando un caso de prueba no cumple el resultado esperado. Cada defecto debe volver al flujo de trabajo para correccion y posterior revalidacion.

Archivo fuente:

```text
docs/pruebas-software/registro-defectos.csv
```

## Defectos registrados

| ID | Caso | Resumen | Severidad | Estado | Accion |
| --- | --- | --- | --- | --- | --- |
| BUG-001 | TC-FUN-003 | App Android no reconoce igual que script Python | Alta | En Proceso | Alinear preprocesamiento Android |
| BUG-002 | TC-FUN-004 | Predicciones cuando no hay gesto claro | Alta | Mitigado | Agregar clase desconocido y umbrales |
| BUG-003 | TC-IA-002 | Clases con pocas muestras o bajo desempeno | Media | Abierto | Grabar mas muestras y reentrenar |
| BUG-004 | TC-CARGA-001 | Plan JMeter inicial no cargaba correctamente | Media | Cerrado | Corregir JMX y revalidar |
| BUG-005 | TC-W3C-001 | Validacion W3C pendiente | Baja | Pendiente | Generar evidencia o justificar no aplicabilidad |

## Flujo de revalidacion

1. Caso falla.
2. Defecto se registra.
3. Item vuelve a `To Do`.
4. Responsable corrige.
5. Caso pasa a `Revalidacion`.
6. Si pasa, se mueve a `Aprobada`.
7. Si falla, se conserva como `Fallida` y vuelve a `To Do`.

## Relacion con prototipado evolutivo

Los defectos no son solo errores aislados: alimentan el siguiente prototipo. Por ejemplo, BUG-001 genero una mejora tecnica en Android para usar luminancia YUV y acercar la app al comportamiento del script Python.
