# Backend

Este modulo queda preparado para una API futura. En el prototipo actual, la app Android y los scripts de IA funcionan sin backend obligatorio.

Para la actividad academica de pruebas de carga se incluye un servidor HTTP local:

```powershell
python backend\load_test_server.py --host 127.0.0.1 --port 8080
```

Endpoints disponibles para JMeter:

- `GET /api/health`
- `GET /api/classes`
- `GET /api/model-version`
- `GET /api/samples/count`
- `POST /api/samples`

Plan de JMeter:

```text
backend/jmeter/signvoice_load_test.jmx
```

Guia de ejecucion y sustentacion:

```text
docs/PRUEBAS_CARGA_JMETER.md
```

Posibles responsabilidades futuras:

- Gestion de usuarios y sesiones.
- Registro de predicciones anonimizadas.
- Versionamiento remoto de modelos.
- Panel de evaluacion academica.
- Servicio de traduccion o auditoria de resultados.
