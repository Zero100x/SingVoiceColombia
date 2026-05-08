# Guia de Contribucion

## Flujo de ramas

- `main`: version estable y entregable.
- `develop`: integracion principal de cambios aprobados.
- `feature/*`: nuevas funcionalidades.
- `prototype/*`: experimentos e iteraciones del prototipo evolutivo.
- `hotfix/*`: correcciones urgentes sobre versiones estables.

## Convencion de commits

Usar mensajes claros en estilo Conventional Commits:

```text
feat: add alphabet recognition service
fix: correct dataset path resolution
docs: document prototype iteration v0.2
chore: update gitignore for Android and ML artifacts
```

## Flujo sugerido

```powershell
git checkout develop
git checkout -b feature/nombre-corto
git add .
git commit -m "feat: describe cambio"
git push -u origin feature/nombre-corto
```

Abrir pull request hacia `develop`. Las versiones estables se integran de `develop` a `main`.

## Reglas de cuidado

- No subir datasets completos ni modelos pesados.
- No subir `local.properties`, `.env`, `.venv`, `.gradle` ni builds.
- Documentar cambios estructurales en `CHANGELOG.md`.
- Mantener compatibilidad con Windows.
