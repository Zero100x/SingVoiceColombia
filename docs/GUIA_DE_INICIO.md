# SingVoiceColombia - Guia de inicio

Esta guia resume como empezar a trabajar con el proyecto `traductor_lsc`.

El repositorio tiene dos partes principales:

- `app_android`: aplicacion Android en Kotlin para reconocimiento del abecedario LSC con camara, texto y voz.
- `scripts`: flujo experimental en Python para capturar imagenes, extraer caracteristicas, entrenar un modelo y probarlo desde computador.

## Estado actual del proyecto

La version Android actual ya no es una carpeta pendiente. Es la aplicacion principal del proyecto.

Actualmente incluye:

- pantalla principal con vista de camara
- solicitud del permiso de camara
- analisis de imagenes con CameraX
- servicio preparado para cargar un modelo TensorFlow Lite
- lectura de etiquetas desde `assets/labels.txt`
- salida de voz para pronunciar la letra detectada

Punto importante: la app Android espera encontrar este archivo:

```text
traductor_lsc/app_android/app/src/main/assets/modelo_alfabeto.tflite
```

Si ese modelo no existe, la app compila, pero mostrara que el modelo esta pendiente y no realizara predicciones reales.

## Estructura general

```text
traductor_lsc/
|- GUIA_DE_INICIO.md
|- app_android/
|  |- app/
|  |  |- build.gradle
|  |  |- src/main/
|  |     |- AndroidManifest.xml
|  |     |- assets/
|  |     |  |- labels.txt
|  |     |- java/com/singvoicecolombia/datasetcapture/
|  |     |  |- activities/
|  |     |  |  |- MainActivity.kt
|  |     |  |- services/
|  |     |  |  |- AlphabetRecognitionService.kt
|  |     |  |- utils/
|  |     |     |- PermissionHelper.kt
|  |     |     |- VoiceOutputManager.kt
|  |     |- res/
|  |        |- drawable/
|  |        |- layout/
|  |        |- values/
|  |        |- xml/
|  |- build.gradle
|  |- gradlew
|  |- gradlew.bat
|  |- settings.gradle
|- dataset/
|- modelos/
|- scripts/
   |- capturar_dataset.py
   |- detectar_mano.py
   |- extraer_caracteristicas.py
   |- entrenar_modelo.py
   |- usar_modelo.py
```

## Requisitos

### Para Android

- Android Studio instalado
- SDK de Android configurado
- JDK 17
- dispositivo fisico o emulador con camara
- conexion a internet la primera vez que Gradle descargue dependencias

La app usa:

- Kotlin
- AndroidX
- Material Components
- CameraX
- TensorFlow Lite

### Para Python

- Python 3
- webcam funcional
- entorno virtual recomendado

Instala las librerias principales:

```bash
pip install opencv-python mediapipe numpy scikit-learn scikit-image joblib
```

## Ejecutar la app Android

Desde Android Studio:

1. Abre la carpeta `traductor_lsc/app_android`.
2. Espera la sincronizacion de Gradle.
3. Verifica que exista `local.properties` con la ruta del SDK.
4. Conecta un celular o inicia un emulador con camara.
5. Ejecuta el modulo `app`.
6. Acepta el permiso de camara cuando la app lo solicite.

Desde PowerShell:

```powershell
cd traductor_lsc\app_android
.\gradlew.bat assembleDebug
```

El APK de depuracion queda en:

```text
traductor_lsc/app_android/app/build/outputs/apk/debug/app-debug.apk
```

## Funcionamiento de la app Android

La pantalla principal esta implementada en:

```text
app/src/main/java/com/singvoicecolombia/datasetcapture/activities/MainActivity.kt
```

Flujo principal:

1. El usuario activa la camara.
2. `MainActivity` solicita permiso si hace falta.
3. CameraX abre la vista previa.
4. Cada cierto intervalo se analiza un frame.
5. `AlphabetRecognitionService` intenta clasificar la imagen con TensorFlow Lite.
6. La letra detectada y la confianza se muestran en pantalla.
7. El boton de voz pronuncia la ultima letra detectada.

Archivos clave:

- `MainActivity.kt`: pantalla principal, camara, botones y estado visual.
- `AlphabetRecognitionService.kt`: carga `modelo_alfabeto.tflite`, lee `labels.txt` y ejecuta inferencia.
- `PermissionHelper.kt`: validacion del permiso de camara.
- `VoiceOutputManager.kt`: salida de texto a voz.
- `activity_main.xml`: interfaz principal.
- `labels.txt`: etiquetas del alfabeto reconocible.

## Modelo para Android

El servicio de reconocimiento busca el modelo en los assets de Android:

```text
app/src/main/assets/modelo_alfabeto.tflite
```

Las etiquetas actuales estan en:

```text
app/src/main/assets/labels.txt
```

El orden de `labels.txt` debe coincidir exactamente con el orden de salida del modelo `.tflite`.

Nota: los scripts actuales de Python entrenan un modelo Random Forest y generan archivos `.pkl`. Esos archivos no se pueden usar directamente en Android como TensorFlow Lite. Para reconocimiento real en la app se necesita entrenar o convertir un modelo compatible con `.tflite`.

## Flujo experimental en Python

Los scripts estan pensados para pruebas desde computador. Ejecutalos desde la carpeta `traductor_lsc/scripts` para que las rutas relativas funcionen correctamente.

```powershell
cd traductor_lsc\scripts
```

### 1. Probar deteccion de mano

```bash
python detectar_mano.py
```

Sirve para validar que la webcam funciona y que MediaPipe detecta la mano.

### 2. Capturar dataset

```bash
python capturar_dataset.py
```

El script pide el nombre de la sena o clase, por ejemplo:

```text
a
b
hola
gracias
```

Luego guarda imagenes en:

```text
traductor_lsc/dataset/<nombre_de_la_clase>/
```

Controles:

- `ESPACIO`: capturar imagen
- `ESC`: salir

### 3. Extraer caracteristicas

```bash
python extraer_caracteristicas.py
```

Este script procesa las imagenes de `dataset` y genera:

```text
traductor_lsc/modelos/datos_caracteristicas.pkl
traductor_lsc/modelos/etiquetas.pkl
```

### 4. Entrenar modelo experimental

```bash
python entrenar_modelo.py
```

Genera:

```text
traductor_lsc/modelos/modelo_senna.pkl
traductor_lsc/modelos/encoder_senna.pkl
```

### 5. Probar modelo en tiempo real

```bash
python usar_modelo.py
```

Este paso usa el modelo `.pkl` desde computador. No corresponde todavia al modelo `.tflite` que necesita Android.

## Flujo recomendado de trabajo

Para compilar y probar la app:

```text
1. Abrir app_android en Android Studio
2. Compilar el modulo app
3. Ejecutar en celular o emulador
4. Verificar permiso de camara
5. Confirmar si existe modelo_alfabeto.tflite
```

Para experimentar con IA en computador:

```text
1. Capturar imagenes con capturar_dataset.py
2. Extraer caracteristicas con extraer_caracteristicas.py
3. Entrenar con entrenar_modelo.py
4. Probar con usar_modelo.py
5. Preparar un modelo TensorFlow Lite para integrarlo en Android
```

## Solucion de problemas

### Gradle no compila

Verifica:

- abrir la carpeta correcta: `traductor_lsc/app_android`
- tener Android SDK configurado
- tener JDK 17
- tener internet si es la primera sincronizacion

### La app muestra que el modelo esta pendiente

Falta el archivo:

```text
app/src/main/assets/modelo_alfabeto.tflite
```

Agrega un modelo TensorFlow Lite compatible con las etiquetas de `labels.txt`.

### La camara no abre en Android

Verifica:

- que el permiso de camara este concedido
- que el emulador tenga camara configurada
- que el celular no tenga la camara ocupada por otra app

### Python no encuentra librerias

Instala las dependencias:

```bash
pip install opencv-python mediapipe numpy scikit-learn scikit-image joblib
```

### Python no encuentra el dataset

Ejecuta los scripts desde:

```text
traductor_lsc/scripts
```

Las rutas de los scripts usan `../dataset` y `../modelos`.

### Baja precision del modelo experimental

Recomendaciones:

- capturar mas imagenes por clase
- usar buena iluminacion
- mantener fondo limpio
- incluir variaciones de angulo y distancia
- evitar clases con pocas muestras

## Proximos pasos sugeridos

1. Completar o generar `modelo_alfabeto.tflite`.
2. Asegurar que `labels.txt` tenga las mismas clases y el mismo orden del modelo.
3. Validar la inferencia en Android con imagenes reales de camara.
4. Documentar el proceso exacto de conversion o entrenamiento del modelo TensorFlow Lite.
5. Limpiar o separar los modelos experimentales `.pkl` de los modelos listos para Android.
