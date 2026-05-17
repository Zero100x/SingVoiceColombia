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

## Importacion desde ZIPs externos

Se agrego el script:

```powershell
python ai\scripts\importar_datasets_zip.py --min-frames 30
```

Este script adapta los ZIP ubicados en `datasets/` al formato local del proyecto:

- `LSC70.zip`: importa `LSC70ANH` y `LSC70W` hacia `datasets/static_signs`.
- `Videos Colors.zip`: convierte videos `.avi` hacia `datasets/dynamic_signs/<color>/muestra_###`.
- `Videos Numbers.zip`: convierte videos `.avi` hacia `datasets/dynamic_signs/<numero>/muestra_###`.

Para las muestras dinamicas, cada video se transforma en 30 frames `frame_###.jpg`. Si el video tiene menos de 30 frames, se remuestrea con repeticion controlada para mantener el formato esperado.

Normalizaciones aplicadas:

- Los nombres de clases se convierten a minusculas sin tildes.
- `dies` se guarda como `diez`.
- `LSC70AN` no se importa por defecto porque duplica clases de `LSC70ANH` con imagenes mas grandes.

Conteo despues de la importacion:

```text
LSC70 estatico: 19704 imagenes importadas.
Videos Colors: 1110 muestras dinamicas importadas.
Videos Numbers: 542 muestras dinamicas importadas.
```

## Buenas practicas de captura

- Mantener iluminacion constante.
- Usar fondo simple.
- Centrar la mano en el encuadre.
- Capturar multiples variaciones por clase.
- Registrar fecha, dispositivo y condiciones de captura.

## Versionamiento de datos

Los datos completos no se suben a Git. Para evidencias academicas, documentar conteos y muestras representativas en `docs/` o `resources/`.
