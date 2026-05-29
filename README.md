# DetectorCam - modo seguro

Version corregida para evitar bloqueos en celulares.

Cambios principales:
- Carga primero el modelo de cuerpo/pose, que es el mas importante para ver esqueleto rapido.
- Manos y cara se cargan solo si eliges el modo completo.
- El modo por defecto ahora es Solo cuerpo rapido.
- Se evita cargar objetos al inicio para no congelar navegadores de celular.
- La camara puede abrir aunque la IA aun este cargando.
- Service Worker actualizado con cache nueva: detectorcam-v7-modo-seguro.

Prueba recomendada:
1. Sube los archivos a GitHub Pages.
2. Abre la app con ?v=modoseguro1.
3. Toca Abrir camara.
4. Espera a que diga IA lista: cuerpo.
5. Prueba primero en Solo cuerpo rapido.
6. Solo despues prueba Cuerpo + manos o Cuerpo + manos + cara.