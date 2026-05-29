# DetectorCam

Aplicación web PWA para detectar movimiento y marcas corporales con MediaPipe.

## Requisitos

- Node.js 18 o superior.
- Navegador moderno basado en Chromium, Edge o Chrome recomendado.
- Permiso de cámara.
- Internet para cargar MediaPipe y modelos de IA.

## Cómo abrir

Opción 1, Windows:

1. Abre `abrir_detectorcam.bat`.
2. El navegador abre `http://127.0.0.1:8787`.

Opción 2, terminal:

```bash
npm start
```

Luego abre:

```text
http://127.0.0.1:8787
```

## Cambios de esta versión

- Se corrigió el `.bat` para que funcione en otros computadores con Node.js instalado.
- La cámara ya no pide micrófono al abrir.
- El audio se solicita solo al iniciar grabación. Si el usuario no lo permite, graba sin audio.
- Se bloquea reiniciar o cambiar cámara mientras se está grabando.
- Se fijó la version de MediaPipe en `0.10.22` para evitar cambios inesperados de `latest`.
- Se ajustó el service worker para cachear solo recursos locales.
- Se agregó `package.json` con comandos `npm start` y `npm run check`.

## Validación técnica

```bash
npm run check
```
