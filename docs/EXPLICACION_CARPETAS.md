# Explicacion de Carpetas del Proyecto

Este archivo resume que hace cada carpeta importante del repositorio `SingVoiceColombia`, con base en la estructura real del proyecto.

## 1. Carpetas de la raiz

### `.git`
Contiene el historial de versiones del proyecto administrado con Git.

### `.idea`
Guarda configuraciones locales de IntelliJ IDEA o Android Studio.
No contiene logica del sistema.

### `.venv`
Entorno virtual de Python.
Se usa para instalar librerias necesarias para los scripts de vision por computador y entrenamiento.

### `.venv-1`
Segundo entorno virtual de Python creado localmente.
Tambien es parte del entorno de trabajo, no del codigo funcional.

### `traductor_lsc`
Es la carpeta principal del desarrollo.
Aqui estan la app Android, el dataset local, los modelos y los scripts de IA.

## 2. Carpetas dentro de `traductor_lsc`

### `app_android`
Contiene la aplicacion movil Android desarrollada en Kotlin.
Es el modulo principal que implementa login, roles, captura del dataset, traduccion simulada, progreso y almacenamiento local.

### `dataset`
Guarda imagenes del dataset usadas por los scripts de Python.
Actualmente contiene una carpeta `hola` con imagenes de ejemplo.

### `modelos`
Reservada para los archivos generados por el entrenamiento del modelo.
Aqui se guardan archivos como caracteristicas extraidas, etiquetas, modelo entrenado y encoder.
En el estado actual puede estar vacia hasta ejecutar los scripts de entrenamiento.

### `scripts`
Contiene scripts en Python para experimentacion con vision por computador e inteligencia artificial.
Aqui estan los procesos de captura, deteccion de mano, extraccion de caracteristicas, entrenamiento y uso del modelo.

## 3. Carpetas dentro de `traductor_lsc/app_android`

### `.gradle`
Carpeta generada automaticamente por Gradle.
Guarda cache y datos internos de compilacion.
No hace parte de la logica del sistema.

### `app`
Es el modulo Android principal.
Aqui estan el codigo fuente, los recursos visuales, el manifiesto y la configuracion de la aplicacion.

### `gradle`
Contiene el wrapper de Gradle.
Permite compilar el proyecto sin depender de una instalacion global manual de Gradle.

## 4. Carpetas dentro de `traductor_lsc/app_android/app/src/main`

### `java`
Contiene el codigo fuente Kotlin de la aplicacion.
La logica del sistema esta organizada por paquetes.

### `res`
Contiene los recursos visuales y de configuracion de Android.
Aqui estan los layouts XML, colores, textos, iconos y archivos XML auxiliares.

## 5. Carpetas dentro de `java/com/singvoicecolombia/datasetcapture`

### `activities`
Contiene las pantallas principales de la app.
Cada actividad controla una parte del flujo de usuario, por ejemplo:
- login
- registro
- menu principal
- captura de dataset
- camara
- traduccion
- progreso
- terminos
- acerca del proyecto

### `adapters`
Contiene adaptadores para mostrar informacion en listas o recyclerviews.
En este proyecto se usa para renderizar el progreso del dataset.

### `auth`
Contiene la logica de autenticacion y control de acceso.
Aqui se manejan:
- usuarios
- sesiones
- roles
- almacenamiento local de usuarios
- validacion de acceso a funciones de administrador

### `data`
Existe como paquete, pero actualmente no tiene implementacion activa.
Puede quedar reservado para una futura capa de datos o repositorios.

### `models`
Contiene las clases de datos del sistema.
Representa objetos como:
- usuario
- sesion
- entrada del dataset
- significado de una sena
- muestra a traducir
- resultado de traduccion

### `services`
Contiene servicios reutilizables de negocio.
En este proyecto incluye la interfaz de traduccion y la implementacion simulada que devuelve resultados sin usar aun un modelo de IA real embebido en Android.

### `utils`
Contiene utilidades compartidas por varias pantallas.
Aqui estan piezas importantes como:
- `DatasetStorageManager`, que organiza y guarda el dataset local
- `PermissionHelper`, que verifica permisos de camara
- `VoiceOutputManager`, que convierte texto a voz

## 6. Carpetas dentro de `res`

### `drawable`
Contiene recursos graficos, fondos, iconos y formas reutilizables para la interfaz.

### `layout`
Contiene los archivos XML de cada pantalla y de algunos componentes visuales.

### `mipmap-anydpi-v26`
Contiene recursos del icono de la aplicacion para Android.

### `values`
Contiene configuraciones globales como:
- colores
- textos
- temas

### `xml`
Contiene archivos XML de configuracion del sistema Android, por ejemplo:
- rutas para `FileProvider`
- reglas de respaldo
- reglas de extraccion de datos

## 7. Explicacion de la carpeta `scripts`

### `capturar_dataset.py`
Captura imagenes desde la camara del computador y las guarda por clase o sena.

### `detectar_mano.py`
Usa MediaPipe para detectar la mano en tiempo real.
Sirve como prueba base de vision por computador.

### `extraer_caracteristicas.py`
Procesa las imagenes del dataset y extrae caracteristicas visuales.
Combina HOG y histogramas de color.

### `entrenar_modelo.py`
Entrena un modelo de clasificacion con los datos extraidos.
Actualmente usa Random Forest.

### `usar_modelo.py`
Carga el modelo entrenado y realiza predicciones en tiempo real desde la camara.

## 8. Resumen rapido para exponer

Si necesitas decirlo de forma corta en la exposicion, puedes resumirlo asi:

- `traductor_lsc/app_android`: aplicacion Android principal.
- `traductor_lsc/scripts`: parte experimental de IA y vision por computador.
- `traductor_lsc/dataset`: imagenes crudas del dataset para entrenamiento.
- `traductor_lsc/modelos`: archivos generados del modelo entrenado.
- `activities`, `auth`, `models`, `services`, `utils`: organizan la logica interna de la app.
- `res`: contiene toda la parte visual y de configuracion Android.

## 9. Nota importante

En la app Android actual, la traduccion dentro del celular todavia es simulada.
La arquitectura ya deja preparado el punto de integracion para conectar despues un modelo real de IA.
