# Dataset de Sennas

## Ubicacion local

```text
datasets/
|- static_signs/
|- dynamic_signs/
|- archives/
```

## Dataset estatico

Uso esperado:

```text
datasets/static_signs/
|- a/
|  |- a_001.jpg
|- b/
|  |- b_001.jpg
```

Sirve para alfabeto, palabras o gestos sin movimiento relevante.

## Dataset dinamico

Uso esperado:

```text
datasets/dynamic_signs/
|- hola/
|  |- muestra_001/
|  |  |- frame_001.jpg
|  |  |- frame_002.jpg
```

Sirve para sennas que requieren movimiento temporal.

## Buenas practicas de captura

- Mantener iluminacion constante.
- Usar fondo simple.
- Centrar la mano en el encuadre.
- Capturar multiples variaciones por clase.
- Registrar fecha, dispositivo y condiciones de captura.

## Versionamiento de datos

Los datos completos no se suben a Git. Para evidencias academicas, documentar conteos y muestras representativas en `docs/` o `resources/`.
