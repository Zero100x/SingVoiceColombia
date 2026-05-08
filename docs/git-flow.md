# Git Flow

## Ramas

- `main`: contiene versiones estables y etiquetas semanticas.
- `develop`: integra trabajo listo para pruebas.
- `feature/*`: nuevas funcionalidades.
- `prototype/*`: experimentos de prototipado evolutivo.
- `hotfix/*`: correcciones urgentes.

## Versionamiento

```text
v0.1.0 -> prototipo inicial
v0.2.0 -> mejoras de interfaz
v0.3.0 -> reconocimiento funcional
v0.4.0 -> sennas dinamicas
v0.5.0 -> validacion academica
```

## Comandos utiles

```powershell
git checkout develop
git checkout -b feature/nueva-funcionalidad
git add .
git commit -m "feat: describe nueva funcionalidad"
git push -u origin feature/nueva-funcionalidad
```

Para crear una version estable:

```powershell
git checkout main
git merge develop
git tag -a v0.1.0 -m "v0.1.0 prototipo inicial"
git push origin main develop --tags
```
