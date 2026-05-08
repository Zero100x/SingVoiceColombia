# SingVoiceColombia

SingVoiceColombia es un proyecto de grado orientado a la traduccion del Lenguaje de Sennas Colombiano (LSC) mediante reconocimiento por inteligencia artificial y vision por computador. El repositorio esta organizado como un producto en evolucion: app movil Android, scripts de IA, datasets locales, documentacion tecnica y bitacora de prototipos.

## Objetivo

Construir un prototipo evolutivo capaz de capturar muestras de sennas, entrenar modelos de reconocimiento y traducir resultados hacia texto y voz dentro de una aplicacion movil.

## Problematica

La comunicacion entre personas usuarias de LSC y personas oyentes puede verse limitada por la falta de herramientas accesibles, locales y adaptadas al contexto colombiano. Este proyecto busca explorar una solucion academica que ayude a reducir esa brecha mediante captura de datos, reconocimiento visual y traduccion asistida.

## Tecnologias

- Android nativo con Kotlin y XML.
- CameraX para captura y analisis de camara.
- TensorFlow Lite para inferencia movil.
- Python para captura, procesamiento y entrenamiento.
- OpenCV, NumPy, scikit-learn, scikit-image y TensorFlow.
- Git Flow con ramas `main`, `develop`, `feature/*`, `hotfix/*` y `prototype/*`.

## Arquitectura General

```text
Camara / dataset local
        |
        v
Preprocesamiento con OpenCV
        |
        v
Entrenamiento de modelos IA
        |
        v
Exportacion TensorFlow Lite
        |
        v
App Android: reconocimiento, texto y voz
```

## Metodologia: Prototipado Evolutivo

El proyecto se desarrolla por iteraciones funcionales. Cada version agrega capacidades medibles, permite validar el prototipo con datos reales y deja preparada la siguiente mejora.

| Version | Hito | Enfoque |
| --- | --- | --- |
| v0.1 | Prototipo inicial | Estructura base, captura de dataset y documentacion inicial. |
| v0.2 | Mejoras de interfaz | Flujo Android mas claro para captura, progreso y traduccion simulada. |
| v0.3 | Reconocimiento funcional | Integracion de modelo TFLite para alfabeto LSC. |
| v0.4 | Sennas dinamicas | Captura secuencial y reconocimiento temporal. |
| v0.5 | Validacion academica | Pruebas, metricas, ajustes y evidencias del proyecto de grado. |

## Estructura del Proyecto

```text
SingVoiceColombia/
|- ai/
|  |- models/              # Artefactos generados y metricas del modelo
|  |- scripts/             # Captura, entrenamiento e inferencia
|- backend/                # Preparado para futuras APIs o servicios
|- datasets/               # Datasets locales ignorados por Git
|  |- static_signs/
|  |- dynamic_signs/
|  |- archives/
|- docs/                   # Documentacion tecnica y diagramas
|- frontend/
|  |- android/             # Aplicacion Android nativa
|- resources/              # Capturas, imagenes y material de apoyo
|- wiki/                   # Wiki lista para GitHub
|- CHANGELOG.md
|- CONTRIBUTING.md
|- LICENSE
|- README.md
```

## Instalacion

1. Clonar el repositorio:

```powershell
git clone https://github.com/Zero100x/SingVoiceColombia.git
cd SingVoiceColombia
```

2. Crear entorno virtual de Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Abrir la app Android:

```text
frontend/android
```

En Android Studio, esperar la sincronizacion de Gradle y configurar el SDK local.

## Ejecucion

### Scripts de IA

```powershell
.\.venv\Scripts\Activate.ps1
python ai\scripts\capturar_dataset.py
python ai\scripts\extraer_caracteristicas.py
python ai\scripts\entrenar_modelo.py
python ai\scripts\usar_modelo.py
```

Para el modelo movil:

```powershell
python ai\scripts\entrenar_modelo_tflite.py
python ai\scripts\usar_modelo_tflite.py
```

### Frontend Android

```powershell
cd frontend\android
.\gradlew.bat assembleDebug
```

Luego ejecutar desde Android Studio o instalar el APK generado en un emulador/dispositivo con camara.

### Backend

El backend esta reservado para futuras integraciones. En el estado actual no hay servicio de API obligatorio para ejecutar el prototipo.

## Estado Actual

- App Android con flujo de traduccion preparada para modelo, captura de dataset y salida por voz.
- Scripts Python para captura de sennas estaticas y dinamicas.
- Entrenamiento de modelo clasico y modelo CNN exportable a TensorFlow Lite.
- Estructura profesional preparada para versionamiento, wiki, documentacion y trabajo colaborativo.

## Iteraciones del Prototipo

Las iteraciones se documentan en:

- [CHANGELOG.md](CHANGELOG.md)
- [docs/prototipado-evolutivo.md](docs/prototipado-evolutivo.md)
- [wiki/Bitacora-de-avances.md](wiki/Bitacora-de-avances.md)

## Screenshots

Las capturas de pantalla deben ubicarse en `resources/screenshots/`.

```text
resources/screenshots/
|- home.png
|- capture.png
|- recognition.png
```

## Autores

- Diego / Zero100x - Desarrollo del proyecto de grado SingVoiceColombia.

## Licencia

Este proyecto se distribuye bajo licencia MIT. Ver [LICENSE](LICENSE).
