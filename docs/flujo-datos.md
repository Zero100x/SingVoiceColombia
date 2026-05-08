# Flujo de Datos

## Captura

1. El usuario selecciona una clase o senna.
2. La camara captura imagenes estaticas o secuencias dinamicas.
3. Las muestras se guardan localmente en `datasets/static_signs` o `datasets/dynamic_signs`.

## Preparacion

1. Los scripts leen el dataset local.
2. Se validan imagenes y clases.
3. Se aplica recorte de region de interes, escala de grises, bordes o HOG segun el modelo.

## Entrenamiento

1. Se separan datos de entrenamiento, validacion y prueba.
2. Se entrena un modelo clasico o CNN.
3. Se generan metricas, matriz de confusion y reporte de clasificacion.

## Exportacion

1. El modelo entrenado se exporta a TensorFlow Lite.
2. Las etiquetas se guardan en `frontend/android/app/src/main/assets`.
3. La app Android carga el modelo y procesa frames de camara.

## Inferencia

1. La camara entrega frames.
2. El servicio de reconocimiento preprocesa la imagen.
3. El modelo devuelve probabilidades por clase.
4. La app muestra texto y puede reproducir voz.
