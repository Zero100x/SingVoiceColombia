# Instalacion y Configuracion

## Requisitos

- Windows 10/11.
- Git.
- Python 3.10 o superior.
- Android Studio.
- SDK Android configurado.
- Dispositivo o emulador con camara.

## Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Android

1. Abrir `frontend/android` en Android Studio.
2. Verificar `local.properties` con la ruta del SDK.
3. Sincronizar Gradle.
4. Ejecutar `app` en emulador o dispositivo.

## Compilacion por consola

```powershell
cd frontend\android
.\gradlew.bat assembleDebug
```
