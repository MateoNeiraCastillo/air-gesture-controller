# Spike: mando NES invisible

**Pregunta:** ¿puede una webcam normal + MediaPipe distinguir un mando NES imaginario sostenido con las dos manos: cruceta con el pulgar izquierdo y botón A con el derecho?

**Respuesta (sesión 1):** sí, con calibración al arrancar y CPU. Pendiente: medir la deriva entre sesiones.

## Montaje

- MacBook Pro M1, webcam integrada, 640×480 a 30 fps, luz artificial.
- MediaPipe 1.1.1, `HandLandmarker` en modo VIDEO, `num_hands=2`, umbrales 0,5.
- 6 clips de 30 s, cada uno con una única postura mantenida: `reposo`, `boton_a_mantenido`, `boton_a_ritmo`, `dpad_izq`, `dpad_der`, `combo` (derecha mantenida + A a ritmo).
- Pipeline: `record.py` (vídeo crudo) → `extract.py` (landmarks a `.npz`) → `analyze.py` (métricas y gráficas).

## Resultados

| Métrica | Resultado | Objetivo |
|---|---|---|
| Frames con dos manos (CPU) | 98,0 % | ≥ 95 % |
| Botón A, reposo vs pulsado (calibrado) | 100 % | ≥ 95 % |
| `boton_a_ritmo`: pulsaciones detectadas | 30 en 29 s | ~1/s |
| `combo`: pulsaciones de A detectadas | 29 en 29 s | ~1/s |
| Cruceta, reposo / izq / der (calibrada) | 99,1 % | ≥ 95 % |
| `combo`: cruceta leída como "derecha" | 100 % | ≥ 95 % |
| Inferencia CPU en M1 (p50 / p95) | 38 / 39 ms | ≤ 35 ms |

Calibración: los primeros 90 frames válidos (~3 s) de cada postura colocan los centroides de un clasificador por centroide; se evalúa con el resto del clip. `ritmo` y `combo` no se usan para calibrar nada: son la prueba fuera de muestra.

## Lo que aprendimos (y lo que salió mal)

1. **La GPU estropea la detección.** Confirmado en el M1 con `compare_delegates.py`, sobre los mismos vídeos:

   | Clip | CPU: 2 manos | GPU: 2 manos | GPU: duplicadas | CPU p50 | GPU p50 |
   |---|---|---|---|---|---|
   | `reposo` | 100 % | 6 % | 13 % | 38 ms | 12 ms |
   | `boton_a_mantenido` | 100 % | 39 % | 0 % | 38 ms | 10 ms |
   | `dpad_der` | 88 % | 76 % | 0 % | 38 ms | 7 ms |

   La GPU es ~4 veces más rápida pero inutilizable. **Decisión: CPU.** Habrá que volver a probarla con futuras versiones de MediaPipe.
2. **Los ángulos del pulgar no sirven; la posición sí.** Con el pulgar doblado sobre el puño, MediaPipe estima mal la articulación: había frames claramente pulsados con el ángulo de reposo. La posición de la punta respecto a la base del índice y a la muñeca, normalizada por el tamaño de la mano, sí separa.
3. **En el botón A solo cuenta la altura de la punta.** Las componentes horizontales cambian si la mano gira un poco entre clips, y llevaron al clasificador a detectar 0 pulsaciones en `ritmo`. Usando solo las verticales: 30 de ~29.
4. **En esta postura, la cruceta se ve sobre todo en vertical.** Con el puño vertical y el pulgar arriba, "izquierda" y "derecha" mueven la punta sobre todo en vertical. Solo con dx, derecha vs reposo daba 55 %; con las 4 features, 99 %.
5. **La calibración tiene que esperar a ver la mano.** En `dpad_der`, la mano izquierda no se detectó durante los primeros segundos. Calibrar con "los primeros 3 s" dejaba la clase vacía, así que se calibra con los primeros N frames válidos.

## Limitaciones

- Una sola sesión y una sola luz: no sabemos cuánto cambian los centroides si te quitas y vuelves a poner las manos, ni de un día para otro.
- `dpad_der` tuvo un 12 % de frames sin la mano izquierda con CPU.

## Siguiente

1. Segunda sesión con otra luz → la sección "entre sesiones" de `analyze.py` dice si basta una calibración o hay que recalibrar cada vez.
2. Latencia: con ~38 ms por frame, el bucle no llega a los 30 fps de la cámara. La captura tiene que quedarse siempre con el frame más reciente (hilo de captura que descarta los viejos); si no, los frames se acumulan en el buffer y la latencia crece. Medir la latencia de punta a punta con el juego corriendo (Fase 3).
