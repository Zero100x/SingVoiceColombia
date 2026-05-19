# Pruebas de Software

Las pruebas del proyecto se documentan siguiendo buenas practicas alineadas con **ISO/IEC/IEEE 29119**.

## Objetivo

Garantizar calidad, estabilidad, seguridad y correcto funcionamiento de SignVoice Colombia.

## Paquete de pruebas

```text
docs/pruebas-software/
```

Incluye:

- Plan de pruebas.
- Casos de prueba.
- Matriz de trazabilidad.
- Registro de defectos.
- Evidencias.
- Guion de sustentacion.

## Tipos de prueba cubiertos

| Tipo | Aplicacion en el proyecto |
| --- | --- |
| Exploratoria | Uso libre de app y scripts para encontrar fallos |
| Funcional | Camara, modelo, audio, permisos y traduccion |
| Usabilidad | Claridad de interfaz, estados y tiempo de lectura |
| Accesibilidad | Contraste, tamanos, descripciones de botones |
| W3C | Validacion de reportes HTML o recursos web aplicables |
| Disponibilidad | Endpoint `/api/health` |
| Latencia | Tiempos de respuesta en JMeter |
| Tracert | Conectividad hacia GitHub o servicio remoto |
| Rendimiento | Entrenamiento, inferencia y tiempos |
| Carga | Usuarios concurrentes con JMeter |
| Compatibilidad | Android real y scripts Windows |
| Seguridad basica | Permisos, secretos y configuracion |
| IA | Accuracy, matriz de confusion y clase desconocido |

## Flujo de defectos

Si una prueba falla:

1. Se registra el defecto.
2. Se crea issue o tarea en la plataforma.
3. Se mueve a `To Do`.
4. Se corrige.
5. Se ejecuta revalidacion.
6. Se actualiza evidencia y estado.

## Estados de prueba

- Pendiente.
- En Proceso.
- Aprobada.
- Fallida.
- Bloqueada.
- Revalidacion.
