# DetectorCam v5 Fix IA y Cámara

Versión corregida para resolver el bloqueo en **IA cargando...** y mantener visible el esqueleto cuando hay detección real.

## Cambios principales

- MediaPipe fijado en `@mediapipe/tasks-vision@0.10.15`.
- La IA carga por módulos: cuerpo y manos primero; rostro y objetos después.
- El detector de objetos ya no bloquea el estado **IA lista**.
- Se quitó el uso obligatorio de GPU para evitar bloqueos en celulares.
- La cámara abre aunque la IA tarde o falle.
- La cámara ya no pide micrófono al iniciar.
- El micrófono se pide solo al grabar. Si falla, graba sin audio.
- Botones de cámara bloqueados mientras graba.
- Filtro de movimiento balanceado para reducir falsos positivos sin ocultar el esqueleto.
- Service Worker actualizado para evitar que el celular use una versión vieja.

## Cómo ejecutar

```bash
npm run check
npm start
```

Luego abre:

```text
http://127.0.0.1:8787
```

También puedes usar:

```text
abrir_detectorcam.bat
```

## Prueba recomendada

1. Abre la app.
2. Espera a que diga **IA lista**. Si tarda, igual puedes tocar **Abrir cámara**.
3. Toca **Abrir cámara**.
4. Apunta a un objeto quieto durante 5 segundos.
5. Debe quedar como quieto o sin evento fuerte.
6. Mueve una mano frente a la cámara.
7. Debe aparecer el esqueleto o marcas de mano.
8. Camina frente a la cámara.
9. Debe marcar persona en movimiento.
10. Graba 5 segundos y revisa que el video tenga las marcas.

## Nota para celular

Si el celular sigue mostrando una versión vieja:

1. Abre la configuración del sitio en el navegador.
2. Borra datos del sitio.
3. Cierra la pestaña.
4. Abre de nuevo la app.

Para usar cámara desde otro dispositivo, el navegador exige **HTTPS** o un origen seguro. En el mismo equipo funciona con `localhost` o `127.0.0.1`.
