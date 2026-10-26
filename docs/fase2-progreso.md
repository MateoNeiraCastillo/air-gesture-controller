# Fase 2 — Lógica del controlador (estado al 26-10-2026)

**Objetivo:** convertir lo aprendido en el spike en la lógica del mando: de landmarks por frame a pulsaciones fiables, testeable sin cámara. La integración con el juego (teclas, FullScreenMario) es la Fase 3.

## Hecho

| Paso | Módulo | Qué hace |
|---|---|---|
| 2.1 | `buttons.py`, `config.py`, `config.toml` | Botones IZQ/DER/A; toda la configuración en TOML, validada al arrancar (una errata o un valor fuera de rango impiden arrancar) |
| 2.2 | `calibration.py` | 4 posturas (reposo, A pulsado, cruceta izq., cruceta der.) con frames válidos, medianas y control de calidad |
| 2.3 | `classifier.py` | Botón A: centroide más cercano (altura del pulgar). Cruceta: dirección = objetivo calibrado más cercano; pulsado si avanza ≥ 30 % del recorrido desde el reposo |
| 2.4 | `posture.py` | Avisa si una mano sale del rango de tamaños visto al calibrar durante > 0,5 s (mano girada o a otra distancia) |
| 2.5 | `state_machine.py` | Pulsar/soltar tras 80 ms seguidos; sin mano = sin información (mantiene); > 200 ms sin mano → suelta sus botones |
| — | `capture.py` | Cámara en un hilo que se queda con el frame más reciente (MediaPipe tarda ~38 ms y la cámara da uno cada ~33 ms) |
| — | `scripts/calibrate_live.py` | Calibración en directo + vista de lo que recibiría el juego |
| 2.6 | `guided.py`, `scripts/guided_session.py`, `scripts/evaluate_guided.py` | Sesión guiada de ~2,5 min (3 bloques iguales) y su evaluación. **Escrito y probado sin cámara; falta grabar la sesión real.** |

69 tests pasan.

## Decisiones y hallazgos

1. **El pulgar izquierdo se movía al pulsar A.** En las grabaciones del spike, mientras se pulsaba A, el pulgar izquierdo se colocaba donde se había calibrado "derecha" (noche: 76–110 % del recorrido). En el juego, Mario habría andado a la derecha al saltar. Ningún umbral lo arregla: el gesto de "derecha" era demasiado pequeño y en la misma dirección que "izquierda".
   - **Calibración:** en la postura "A pulsado" también se mira la mano izquierda; si se va hacia una dirección (≥ 15 % del recorrido), se rechaza con un mensaje que dice qué repetir.
   - **Gesto nuevo** (puños verticales, pulgar arriba): reposo = pulgar recto; izquierda = inclinarlo hacia fuera; derecha = hacia dentro, hacia la otra mano; al pulsar A, el pulgar izquierdo recto.
   - **Resultado en directo (2 calibraciones):** aceptadas a la primera; separación derecha–reposo 17 (antes 7–14), izquierda–derecha 23–44; pulgar izquierdo al pulsar A: 0 % hacia derecha (antes 50–52 %).
2. **Umbral de la cruceta: 0,3 del recorrido** (no 0,5). En el `combo` de día el pulgar hacía ~50 % del recorrido calibrado; con 0,5 acertaba el 49 %, con 0,3 el 98 %. En reposo rara vez pasa de 0,2.
3. **La dirección de la cruceta se elige por el objetivo más cercano, no por el mayor avance.** En la noche izquierda y derecha iban en la misma dirección (coseno +0,95) y "derecha" era un eje corto: un punto de "izquierda" avanzaba 3 veces más en ese eje.
4. **El monitor de postura compara con un rango, no con un valor.** La mano se inclina al mover el pulgar (la izquierda iba del 77 % al 130 % de su mediana según la postura, todas válidas). Con un único valor saltaban avisos falsos.
5. **Máquina de estados en el spike:** `ritmo` noche 32 pulsaciones crudas → 30 confirmadas; `combo` 29–30 → 29. En el `ritmo` de día (puño girado) no confirma ninguna, pero el monitor avisa el 76 % del tiempo: falla con aviso, no en silencio.

## Pendiente para cerrar la Fase 2

- Grabar `scripts/guided_session.py` (~3 min) y pasar `scripts/evaluate_guided.py`.
- Criterios: peor botón ≥ 95 % del tiempo correcto; toques de A exactos ≥ 80 %; comparar bloque 1 con bloque 3 (deriva).
- Según el resultado: ajustar `press_confirm_ms` (¿se nota el retraso de 80 ms?) y decidir si hace falta recalibración automática.
