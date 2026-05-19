# Guion de Sustentacion - Pruebas de Software

## Introduccion

Para garantizar la calidad del proyecto SignVoice Colombia se definio un plan de pruebas basado en buenas practicas de la norma ISO/IEC/IEEE 29119. Cada caso de prueba fue documentado con identificador, objetivo, precondiciones, datos de entrada, pasos, resultado esperado, resultado obtenido, estado y evidencia.

## Tipos de pruebas aplicadas

Se contemplaron pruebas exploratorias, funcionales, usabilidad, accesibilidad, disponibilidad, latencia, conectividad con `tracert`, rendimiento, carga, compatibilidad, seguridad basica y pruebas especificas del modelo de inteligencia artificial.

## Pruebas de carga

Se uso Apache JMeter para evaluar endpoints HTTP academicos del proyecto. Se probaron peticiones GET y POST:

- `GET /api/health`
- `GET /api/classes`
- `GET /api/model-version`
- `POST /api/samples`

En una prueba con 50 usuarios concurrentes se registraron 1552 muestras, tiempo promedio total de 554 ms y error total de 0.13%. Esto permitio validar estabilidad basica del servicio local bajo carga.

## Pruebas del modelo IA

El modelo dinamico fue evaluado con reporte de clasificacion y matriz de confusion. La ultima precision de prueba fue 86.61%. Tambien se agrego la clase `DESCONOCIDO` para reducir falsas predicciones cuando el usuario no hace una sena clara.

## Pruebas funcionales en Android

Se valido que la app cargue el modelo, active la camara, muestre estados de reconocimiento, mantenga el resultado visible y pueda reproducir audio automatico. Algunas pruebas quedaron en proceso porque el comportamiento de la app se sigue comparando contra el script Python como parte del prototipado evolutivo.

## Gestion de defectos

Cuando una prueba falla, se registra en el archivo de defectos y debe volver al estado `To Do` o equivalente en el tablero. Esto permite que el desarrollador responsable atienda el bug y luego se ejecute una revalidacion.

## Cierre

La documentacion de pruebas demuestra que el proyecto no solo fue implementado, sino tambien evaluado tecnicamente. Las pruebas permiten identificar riesgos, validar avances y alimentar nuevas iteraciones del prototipado evolutivo.
