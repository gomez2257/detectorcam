# DetectorCam v5 Pro

Versión combinada entre la última actualización con clasificación de eventos y el filtro estricto contra falsos positivos.

## Cómo abrir

Opción 1 en Windows:

1. Instala Node.js si no lo tienes.
2. Ejecuta `abrir_detectorcam.bat`.
3. Abre `http://127.0.0.1:8787`.

Opción 2 por consola:

```bash
npm start
```

## Qué mejora esta versión

- Mantiene la clasificación de eventos: persona, animal, objeto, anomalía y forma quieta.
- Reduce falsos positivos sobre objetos inmóviles como bolsos, almohadas o ropa.
- No marca movimiento solo porque MediaPipe dibuje un esqueleto falso.
- Exige movimiento visual local, desplazamiento de landmarks y varios puntos activos.
- Usa MediaPipe con versión fija `0.10.22`.
- La cámara no pide micrófono al iniciar.
- El micrófono se pide solo al grabar. Si no hay permiso, graba sin audio.
- Bloquea reiniciar o cambiar cámara mientras graba.
- Mejora el service worker para cachear solo archivos locales.

## Validación

Ejecuta:

```bash
npm run check
```

Esto revisa sintaxis de `app.js`, `server.js` y `service-worker.js`.

## Prueba recomendada

1. Apunta la cámara a un objeto quieto.
2. Espera de 3 a 5 segundos.
3. Debe aparecer como forma quieta u objeto quieto, no como movimiento.
4. Mueve una mano o camina frente a la cámara.
5. Debe marcar persona en movimiento o evento detectado.
