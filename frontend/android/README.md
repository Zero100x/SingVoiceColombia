# SingVoiceColombia Android

Aplicacion Android nativa del prototipo SingVoiceColombia. Este modulo contiene la interfaz movil para captura de dataset, reconocimiento preparado con TensorFlow Lite, traduccion a texto y salida por voz.

## Ubicacion

```text
frontend/android
```

## Funcionalidades actuales

- Pantalla principal del prototipo.
- Captura de dataset con CameraX.
- Flujo de reconocimiento preparado para modelos `.tflite`.
- Servicios separados para reconocimiento de alfabeto y sennas dinamicas.
- Salida de voz mediante utilidades Android.

## Ejecucion

1. Abrir esta carpeta en Android Studio.
2. Sincronizar Gradle.
3. Verificar `local.properties` con el SDK Android local.
4. Ejecutar el modulo `app` en emulador o dispositivo fisico.

## Compilacion por consola

```powershell
cd frontend\android
.\gradlew.bat assembleDebug
```

## Assets de IA

Los modelos generados por Python deben ubicarse en:

```text
frontend/android/app/src/main/assets/
|- labels.txt
|- modelo_alfabeto.tflite
|- dynamic_labels.txt
|- modelo_dinamico.tflite
```

Los archivos `.tflite` son artefactos generados y no se versionan por defecto.
