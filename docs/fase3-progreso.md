# Fase 3 — Jugar con el mando (en curso, 27-10-2026)

**Objetivo:** conectar la lógica de la Fase 2 con el juego: teclas reales, bucle completo y latencia medida.

## Hecho (probado sin cámara, con vídeos y teclado simulado; 81 tests)

| Módulo | Qué hace |
|---|---|
| `air_controller/output.py` | Capa 5. Traduce eventos de botón a teclas con pynput. Recuerda qué teclas tiene pulsadas y las suelta siempre: al salir, al pausar, al recalibrar, ante un error o con Ctrl+C. Valida los nombres de tecla al arrancar. `RecordingKeySender` para tests y `--dry-run`. |
| `air_controller/latency.py` | Latencia por etapa y de punta a punta (frame recibido → tecla enviada), p50/p95/máx, y FPS del bucle. |
| `air_controller/capture.py` | Cada frame guarda el instante en que llegó de la cámara. |
| `scripts/play.py` | El controlador completo: calibrar → jugar. P pausa, R recalibra, Q sale. `--dry-run` imprime las teclas en vez de enviarlas. Al salir imprime la latencia. |

**Freno de emergencia:** apartar las manos de la cámara suelta todas las teclas a los 200 ms (`missing_hand_ms`).

## Primera medición real (M1, `play.py --dry-run`, 27-10-2026)

| Etapa | p50 | p95 |
|---|---|---|
| Espera del frame (antigüedad al cogerlo) | 18 ms | 33 ms |
| Detección (MediaPipe, CPU) | 37 ms | 37 ms |
| Clasificar + máquina de estados | 0,1 ms | 0,1 ms |
| **Frame → tecla** | **49 ms** | **56 ms** |
| FPS del bucle | 19,6 | |

- Leer siempre el frame más reciente funciona: la espera media es medio frame (18 ms) y no crece.
- Muy estable: p95 casi igual a p50.
- **No incluye** la confirmación de pulsación (2 frames seguidos: +~50 ms a 19,6 fps) ni el retraso interno de la cámara (~30–60 ms típico). Retraso percibido estimado: **~130–160 ms**, en el límite del presupuesto de 100–120 ms.
- El bucle va a 19,6 fps (~51 ms por vuelta): 37 ms de MediaPipe + ~14 ms de dibujo y ventana. Primera optimización si se nota retraso al jugar: abaratar el dibujo (más barata que tocar MediaPipe).

## Pendiente

1. **Permiso de Accesibilidad** en macOS para la app desde la que se ejecuta (VS Code o Terminal). Sin él, pynput no da error pero las teclas no llegan.
2. **Confirmar la tecla de salto de FullScreenMario.** Según la documentación encontrada, se mueve con flechas o WASD y Shift sirve para correr; el salto no está confirmado. `config.toml` tiene `a = "up"` provisional.
3. **Medir FPS y latencia reales** con `play.py --dry-run` y después jugando. Si el bucle sigue a ~15 fps, optimizar: dibujar menos, o mover MediaPipe a su propio hilo / modo LIVE_STREAM.
4. **Botón B (correr = Shift)**, cuando todo lo anterior funcione.
