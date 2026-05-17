# SingVoiceColombia - Guia de inicio

Esta guia resume los pasos basicos para instalar, ejecutar y alimentar el dataset del proyecto **SingVoiceColombia**.

El repositorio esta organizado en cuatro areas principales:

- `frontend/android`: aplicacion Android nativa en Kotlin para captura, reconocimiento, texto y voz.
- `ai/scripts`: scripts Python para captura, procesamiento, entrenamiento e inferencia.
- `datasets`: datos locales para entrenamiento. Esta carpeta no se sube completa a Git.
- `ai/models`: modelos, metricas y reportes generados localmente.

## Estado actual del proyecto

La aplicacion Android es el frontend principal del prototipo. Actualmente incluye:

- pantalla principal con camara
- flujo de captura de dataset
- reconocimiento preparado con TensorFlow Lite
- servicios separados para alfabeto y sennas dinamicas
- lectura de etiquetas desde `assets`
- salida de voz para pronunciar el resultado detectado

Los modelos `.tflite` se generan desde Python y se copian a:

```text
frontend/android/app/src/main/assets/
|- labels.txt
|- modelo_alfabeto.tflite
|- dynamic_labels.txt
|- modelo_dinamico.tflite
```

Si un modelo no existe, la app puede compilar, pero el reconocimiento real quedara pendiente para ese flujo.

## Estructura general

```text
SingVoiceColombia/
|- ai/
|  |- scripts/
|  |  |- capturar_dataset.py
|  |  |- detectar_mano.py
|  |  |- extraer_caracteristicas.py
|  |  |- entrenar_modelo.py
|  |  |- entrenar_modelo_tflite.py
|  |  |- entrenar_modelo_dinamico.py
|  |  |- usar_modelo.py
|  |  |- usar_modelo_tflite.py
|  |  |- usar_modelo_dinamico.py
|  |- models/
|- backend/
|- datasets/
|  |- static_signs/
|  |- dynamic_signs/
|  |- archives/
|- docs/
|- frontend/
|  |- android/
|- resources/
|- wiki/
|- requirements.txt
```

## Requisitos

### Python e IA

- Python 3.10 o superior recomendado.
- Webcam funcional si vas a capturar desde el computador.
- Entorno virtual de Python.
- Dependencias de `requirements.txt`.

### Android

- Android Studio instalado.
- SDK de Android configurado.
- JDK 17.
- Dispositivo fisico o emulador con camara.
- Internet la primera vez que Gradle descargue dependencias.

La app usa Kotlin, AndroidX, Material Components, CameraX y TensorFlow Lite.

## Preparar entorno Python

Ejecuta estos comandos desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Cuando el entorno este activo, PowerShell mostrara `(.venv)` al inicio de la linea.

Si PowerShell bloquea la activacion del entorno, usa:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Para salir del entorno virtual:

```powershell
deactivate
```

Nota: `ai/scripts/detectar_mano.py` usa MediaPipe para mostrar los puntos de la mano. Primero intenta la API antigua `mp.solutions`; si no existe, usa la API nueva MediaPipe Tasks con `hand_landmarker.task`. Solo usa OpenCV como respaldo si MediaPipe no esta instalado o si falta el archivo `.task`. Si falta la libreria, instala la dependencia adicional:

```powershell
pip install mediapipe
```

## Ejecutar la app Android

Desde Android Studio:

1. Abre la carpeta `frontend/android`.
2. Espera la sincronizacion de Gradle.
3. Verifica que exista `local.properties` con la ruta del SDK.
4. Conecta un celular o inicia un emulador con camara.
5. Ejecuta el modulo `app`.
6. Acepta el permiso de camara cuando la app lo solicite.

Desde PowerShell:

```powershell
cd frontend\android
.\gradlew.bat assembleDebug
```

El APK de depuracion queda en:

```text
frontend/android/app/build/outputs/apk/debug/app-debug.apk
```

## Dataset

El dataset local esta en:

```text
datasets/
|- static_signs/
|- dynamic_signs/
|- archives/
```

Los archivos del dataset no se versionan en Git para evitar un repositorio pesado y proteger capturas locales.

### Sennas estaticas

Usa `datasets/static_signs` para letras, palabras o gestos sin movimiento relevante:

```text
datasets/static_signs/
|- a/
|  |- a_001.jpg
|  |- a_002.jpg
|- b/
|  |- b_001.jpg
```

Tambien puedes capturarlas con:

```powershell
python ai\scripts\capturar_dataset.py
```

Selecciona la opcion:

```text
1 -> Capturar sena estatica
```

Controles:

- `ESPACIO`: capturar imagen.
- `ESC`: salir.

Durante la captura se muestran los puntos de MediaPipe sobre la mano. La imagen se guarda limpia, sin los puntos dibujados. El capturador puede detectar hasta 2 manos, pero por defecto solo exige 1 mano para guardar. Si MediaPipe no detecta suficientes manos, el script no guarda la foto para evitar ensuciar el dataset. Si necesitas guardar de todos modos:

```powershell
python ai\scripts\capturar_dataset.py --permitir-sin-mano
```

### Sennas dinamicas

Usa `datasets/dynamic_signs` para sennas que requieren movimiento:

```text
datasets/dynamic_signs/
|- hola/
|  |- muestra_001/
|  |  |- frame_001.jpg
|  |  |- frame_002.jpg
|  |- muestra_002/
|- atencion/
|  |- muestra_001/
```

Cada muestra debe tener minimo 8 frames validos. Para mejores resultados, intenta tener al menos 16 frames por muestra, 6 muestras por senna como minimo y, si es posible, unas 30 muestras por clase.

Puedes capturarlas directamente con:

```powershell
python ai\scripts\capturar_dataset.py
```

Selecciona la opcion:

```text
2 -> Capturar sena dinamica
```

Al iniciar la captura dinamica, el script pregunta cuantas manos requiere la senna. Para sennas de dos manos, responde `2`. La vista mostrara los puntos de ambas manos y no empezara a grabar si solo detecta una mano.

Tambien puedes dejarlo preparado desde el comando:

```powershell
python ai\scripts\capturar_dataset.py --manos 2 --manos-minimas 2
```

Si solo quieres probar que la camara detecta dos manos:

```powershell
python ai\scripts\detectar_mano.py --manos 2
```

## Convertir un video en frames

Si tienes un video de una senna, debes convertirlo en imagenes `frame_###.jpg` dentro de una carpeta `muestra_###`.

Ejemplo para la senna `atencion`:

```powershell
New-Item -ItemType Directory -Force -Path "datasets\dynamic_signs\atencion\muestra_001"
```

### Opcion 1: con ffmpeg

Si tienes `ffmpeg` instalado:

```powershell
ffmpeg -i "C:\ruta\del\video.mp4" -vf fps=8 "datasets\dynamic_signs\atencion\muestra_001\frame_%03d.jpg"
```

Si PowerShell muestra que `ffmpeg` no se reconoce, usa la opcion con Python/OpenCV.

### Opcion 2: con Python/OpenCV

Este metodo funciona con las dependencias del proyecto:

```powershell
@'
import cv2
from pathlib import Path

video_path = Path(r"datasets\dynamic_signs\atencion\muestra_001\muestra_001.m4v")
out_dir = Path(r"datasets\dynamic_signs\atencion\muestra_001")
target_fps = 8

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit(f"No se pudo abrir el video: {video_path}")

source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, round(source_fps / target_fps))

frame_index = 0
saved = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_index % step == 0:
        saved += 1
        cv2.imwrite(str(out_dir / f"frame_{saved:03d}.jpg"), frame)

    frame_index += 1

cap.release()
print(f"Frames guardados: {saved}")
print(f"Carpeta: {out_dir}")
'@ | python -
```

Cambia `atencion`, `muestra_001` y el nombre del video segun corresponda.

## Flujo de IA en computador

Ejecuta los scripts desde la raiz del proyecto. Las rutas ya estan preparadas para encontrar `datasets` y `ai/models`.

### 1. Probar deteccion de mano

```powershell
python ai\scripts\detectar_mano.py
```

Sirve para validar que la webcam funciona y que MediaPipe detecta la mano.
Si tu version de MediaPipe no trae `mp.solutions`, el script usa MediaPipe Tasks con `hand_landmarker.task` y dibuja los 21 puntos de cada mano detectada. Si tampoco puede cargar Tasks, usa un detector basico con OpenCV para validar camara y contorno de mano.
En Windows, el script prueba varios indices de camara y backends de OpenCV para evitar fallos cuando `VideoCapture(0)` abre el dispositivo pero no entrega frames.

Para forzar una camara especifica:

```powershell
python ai\scripts\detectar_mano.py --camara 1
```

Prueba `--camara 0`, `--camara 1`, `--camara 2` hasta encontrar la camara correcta.

Para detectar dos manos:

```powershell
python ai\scripts\detectar_mano.py --camara 1 --manos 2
```

### 2. Capturar dataset

```powershell
python ai\scripts\capturar_dataset.py
```

Este script permite capturar sennas estaticas y dinamicas.
Muestra los 21 puntos de MediaPipe por cada mano detectada para validar que la captura es correcta antes de guardar.
Para capturar con otra camara:

```powershell
python ai\scripts\capturar_dataset.py --camara 1
```

### 3. Entrenar modelo de alfabeto para Android

```powershell
python ai\scripts\entrenar_modelo_tflite.py
```

Antes de entrenar completo, puedes auditar el dataset sin sobrescribir el modelo:

```powershell
python ai\scripts\entrenar_modelo_tflite.py --solo-auditar
```

Para una auditoria rapida con MediaPipe:

```powershell
python ai\scripts\entrenar_modelo_tflite.py --solo-auditar --limite-por-clase 10
```

El entrenamiento del alfabeto usa MediaPipe Tasks para recortar la mano con `hand_landmarker.task`, igual que la app Android. Si una imagen no tiene mano detectada, se omite para que no ensucie el modelo con cuerpos quietos, caras o fondos. Si necesitas volver al comportamiento anterior:

```powershell
python ai\scripts\entrenar_modelo_tflite.py --recorte-mano centro
```

Lee imagenes desde:

```text
datasets/static_signs/
```

Genera salidas como:

```text
frontend/android/app/src/main/assets/modelo_alfabeto.tflite
frontend/android/app/src/main/assets/labels.txt
ai/models/modelo_alfabeto_metadata.json
```

### 4. Probar modelo de alfabeto

```powershell
python ai\scripts\usar_modelo_tflite.py
```

Este script tambien usa MediaPipe para recortar la mano antes de enviar la imagen al modelo, igual que Android. Para probar el modo anterior de recorte central:

```powershell
python ai\scripts\usar_modelo_tflite.py --recorte-mano centro
```

Para probarlo con otra camara:

```powershell
python ai\scripts\usar_modelo_tflite.py --camara 1
```

El reconocedor puede mostrar `SIN MANO` cuando no detecta una mano dentro del recuadro, o `SIN CONFIANZA` cuando el modelo no diferencia bien la mejor clase de la segunda. Esto evita que siempre fuerce una letra como `A` cuando la imagen no es clara.

### 5. Entrenar modelo dinamico

```powershell
python ai\scripts\entrenar_modelo_dinamico.py
```

Lee secuencias desde:

```text
datasets/dynamic_signs/
```

El entrenamiento dinamico usa 16 frames de cada muestra para crear una entrada compacta de 6 canales:

```text
1. frame inicial en escala de grises
2. bordes del frame inicial
3. frame final en escala de grises
4. bordes del frame final
5. mapa promedio de movimiento
6. bordes del movimiento
```

Las clases con menos de 6 muestras validas se omiten automaticamente para no danar el entrenamiento. Para resultados mas estables, usa al menos 30 muestras por senna dinamica.

Genera salidas como:

```text
frontend/android/app/src/main/assets/modelo_dinamico.tflite
frontend/android/app/src/main/assets/dynamic_labels.txt
ai/models/modelo_dinamico_metadata.json
```

### 6. Probar modelo dinamico

```powershell
python ai\scripts\usar_modelo_dinamico.py
```

Para probarlo con otra camara:

```powershell
python ai\scripts\usar_modelo_dinamico.py --camara 1
```

Si aparece `SIN MANO`, centra mejor la mano en el recuadro y mejora la iluminacion. Si aparece `SIN CONFIANZA`, normalmente falta mas dataset o el modelo necesita reentrenarse con muestras parecidas a la camara actual.

El modelo dinamico espera recolectar 16 frames antes de mostrar una prediccion, asi que realiza el movimiento completo dentro del recuadro verde.

## Flujo recomendado

Para experimentar con una nueva senna dinamica:

```text
1. Crear la carpeta datasets/dynamic_signs/<senna>/muestra_001/
2. Grabar o copiar el video dentro de esa carpeta.
3. Convertir el video a frame_001.jpg, frame_002.jpg, etc.
4. Repetir el proceso con varias muestras.
5. Entrenar con ai/scripts/entrenar_modelo_dinamico.py.
6. Probar con ai/scripts/usar_modelo_dinamico.py.
7. Abrir Android y validar el modelo exportado.
```

Para trabajar con alfabeto o posturas estaticas:

```text
1. Guardar imagenes en datasets/static_signs/<clase>/
2. Entrenar con ai/scripts/entrenar_modelo_tflite.py.
3. Probar con ai/scripts/usar_modelo_tflite.py.
4. Validar en Android.
```

## Buenas practicas de captura

- Mantener iluminacion constante.
- Usar fondo simple.
- Centrar la mano o el cuerpo en el encuadre.
- Evitar videos borrosos o demasiado oscuros.
- Capturar variaciones de persona, distancia y angulo.
- Mantener nombres de carpetas simples: minusculas, numeros, `_` o `-`.
- Documentar procedencia, fecha y condiciones en `docs/dataset.md`.

## Solucion de problemas

### PowerShell no activa el entorno virtual

Ejecuta:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### ffmpeg no se reconoce

Significa que `ffmpeg` no esta instalado o no esta en el `PATH`. Puedes convertir el video con Python/OpenCV usando el ejemplo de esta guia.

### MediaPipe no tiene mp.solutions

En versiones nuevas puede aparecer este error:

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

El script `ai/scripts/detectar_mano.py` ya contempla ese caso y cambia automaticamente a MediaPipe Tasks. Ejecutalo de nuevo:

```powershell
python ai\scripts\detectar_mano.py
```

### La camara no muestra ventana o no entrega frames

Verifica:

- cerrar Zoom, Teams, navegador u otra app que pueda estar usando la camara
- permitir acceso a la camara para aplicaciones de escritorio en Windows
- probar una camara externa o cambiar de puerto USB
- probar otro indice: `python ai\scripts\detectar_mano.py --camara 1`

### Python no encuentra librerias

Activa el entorno e instala dependencias:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Python no encuentra el dataset

Ejecuta los scripts desde la raiz del repositorio:

```powershell
cd "C:\Users\DIEGO\OneDrive\Documents\Universidad\proyecto de grado\SingVoiceColombia"
python ai\scripts\capturar_dataset.py
```

### El entrenamiento dinamico dice que hay pocos frames

Cada `muestra_###` debe tener al menos 8 imagenes con extension `.jpg`, `.jpeg` o `.png`.

Ejemplo valido:

```text
datasets/dynamic_signs/atencion/muestra_001/
|- frame_001.jpg
|- frame_002.jpg
|- frame_003.jpg
|- frame_004.jpg
|- frame_005.jpg
|- frame_006.jpg
|- frame_007.jpg
|- frame_008.jpg
```

### La app muestra que el modelo esta pendiente

Verifica que existan los modelos esperados:

```text
frontend/android/app/src/main/assets/modelo_alfabeto.tflite
frontend/android/app/src/main/assets/modelo_dinamico.tflite
```

Si no existen, entrena el modelo correspondiente desde `ai/scripts`.

### Gradle no compila

Verifica:

- abrir la carpeta correcta: `frontend/android`
- tener Android SDK configurado
- tener JDK 17
- tener internet si es la primera sincronizacion

## Proximos pasos sugeridos

1. Completar mas muestras por cada senna dinamica.
2. Mantener actualizado `docs/dataset.md` con conteos y condiciones de captura.
3. Reentrenar los modelos cuando se agreguen nuevas clases.
4. Validar que `labels.txt` y `dynamic_labels.txt` coincidan con las clases entrenadas.
5. Probar los modelos exportados en Android con videos reales de camara.
