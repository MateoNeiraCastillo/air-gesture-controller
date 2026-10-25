"""Spike paso 3: ¿funciona el mando invisible, y cuánto cambia entre sesiones?

Lee data/landmarks/*.npz (nombre: <etiqueta>_<sesión>_<NN>.npz), clasifica con
la posición del pulgar y un clasificador por centroide, y mide:

  1. Detección por clip.
  2. Por sesión, con CALIBRACIÓN: los primeros CALIBRATION_FRAMES frames válidos
     de cada clip colocan los centroides; se evalúa con el resto del clip.
  3. Entre sesiones (si hay 2 o más): calibrar en una sesión y usar en otra,
     sin recalibrar. Si esto cae mucho, la calibración al arrancar es obligatoria.

Uso:
    python scripts/analyze.py
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # solo guardamos PNG, no abrimos ventanas
import matplotlib.pyplot as plt
import numpy as np

from air_controller.centroid import CentroidClassifier
from air_controller.features import thumb_position_features

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = ROOT / "data" / "landmarks"
REPORT_DIR = ROOT / "data" / "report"

SETTLE_SECONDS = 1.0        # se descarta el primer segundo de cada clip
CALIBRATION_FRAMES = 90     # ~3 s de frames VÁLIDOS (mano detectada) por postura
# Columnas de thumb_position_features: 0 dx_índice, 1 dy_índice, 2 dx_muñeca, 3 dy_muñeca.
# Botón A: solo la altura de la punta. Las dx cambian si la mano gira un poco entre
# clips (se vio en el spike) y arrastraban al clasificador.
BUTTON_FEATURES = [1, 3]
DPAD_FEATURES = [0, 1, 2, 3]
SMOOTHING_FRAMES = 3        # mediana móvil al contar pulsaciones

TWO_HANDS_TARGET = 95.0
ACCURACY_TARGET = 95.0
INFERENCE_P95_TARGET_MS = 35.0

BUTTON_LABELS = ("reposo", "boton_a_mantenido")
DPAD_LABELS = ("reposo", "dpad_izq", "dpad_der")


@dataclass(frozen=True)
class Clip:
    label: str
    session: str
    fps: float
    left: np.ndarray            # (frames, 4) features del pulgar izquierdo
    right: np.ndarray           # (frames, 4) features del pulgar derecho
    hands_detected: np.ndarray
    inference_ms: np.ndarray

    def features(self, hand: str, columns: list[int]) -> np.ndarray:
        return (self.left if hand == "left" else self.right)[:, columns]

    def split(self, hand: str, columns: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(calibración, evaluación): los primeros CALIBRATION_FRAMES frames válidos
        calibran; todo lo posterior evalúa. Como en la app real, la calibración
        espera a ver la mano en vez de usar segundos fijos."""
        features = self.features(hand, columns)
        valid = np.flatnonzero(~np.isnan(features).any(axis=1))
        if len(valid) <= CALIBRATION_FRAMES:
            return features[valid], features[:0]
        cut = valid[CALIBRATION_FRAMES]
        return features[valid[:CALIBRATION_FRAMES]], features[cut:]


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #

def features_per_frame(hands: np.ndarray) -> np.ndarray:
    return np.array([thumb_position_features(hand) for hand in hands]).reshape(-1, 4)


def load_clips() -> dict[str, dict[str, Clip]]:
    """sesión → etiqueta → Clip. Si hay varias tomas iguales, se usa la última."""
    sessions: dict[str, dict[str, Clip]] = defaultdict(dict)
    for path in sorted(LANDMARKS_DIR.glob("*.npz")):
        label, session, _index = path.stem.rsplit("_", 2)
        with np.load(path) as data:
            fps = float(data["fps"])
            settle = int(SETTLE_SECONDS * fps)
            sessions[session][label] = Clip(
                label=label,
                session=session,
                fps=fps,
                left=features_per_frame(data["left"][settle:]),
                right=features_per_frame(data["right"][settle:]),
                hands_detected=data["hands_detected"][settle:],
                inference_ms=data["inference_ms"][settle:],
            )
    return dict(sessions)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #

def accuracy(classifier: CentroidClassifier, features: np.ndarray, expected: str) -> float:
    """% de frames válidos clasificados como `expected` (NaN = mano no vista, no cuenta)."""
    predicted = classifier.predict(features)
    valid = predicted != ""
    return 100 * float(np.mean(predicted[valid] == expected)) if valid.any() else float("nan")


def balanced_accuracy(classifier: CentroidClassifier, clips: dict[str, Clip],
                      labels: tuple[str, ...], hand: str, columns: list[int],
                      after_calibration: bool) -> float:
    """Media del acierto por clase. after_calibration=True: solo frames posteriores a la
    calibración (misma sesión); False: el clip entero (otra sesión)."""
    scores = []
    for label in labels:
        clip = clips[label]
        features = clip.split(hand, columns)[1] if after_calibration else clip.features(hand, columns)
        scores.append(accuracy(classifier, features, label))
    return float(np.nanmean(scores))


def calibrate(clips: dict[str, Clip], labels: tuple[str, ...], hand: str,
              columns: list[int]) -> CentroidClassifier:
    return CentroidClassifier.fit({label: clips[label].split(hand, columns)[0] for label in labels})


def smooth_labels(predicted: np.ndarray, positive: str, frames: int) -> np.ndarray:
    """Booleano 'es positive' con mediana móvil (quita parpadeos de 1 frame)."""
    is_positive = (predicted == positive).astype(float)
    pad = frames // 2
    padded = np.pad(is_positive, pad, mode="edge")
    return np.array([np.median(padded[i:i + frames]) for i in range(len(is_positive))]) > 0.5


def count_presses(pressed: np.ndarray) -> int:
    return int(np.sum(pressed[1:] & ~pressed[:-1]))


def has_all(clips: dict[str, Clip], labels: tuple[str, ...]) -> bool:
    return all(label in clips for label in labels)


# --------------------------------------------------------------------------- #
# Secciones del informe
# --------------------------------------------------------------------------- #

def report_detection(sessions: dict[str, dict[str, Clip]]) -> tuple[float, float]:
    print("\n== Detección por clip ==")
    print(f"{'sesión':<10}{'etiqueta':<20}{'frames':>8}{'2 manos':>10}{'inf p50':>10}{'inf p95':>10}")
    counts, times = [], []
    for session, clips in sorted(sessions.items()):
        for label, clip in sorted(clips.items()):
            counts.append(clip.hands_detected)
            times.append(clip.inference_ms)
            print(f"{session:<10}{label:<20}{len(clip.hands_detected):>8}"
                  f"{100 * np.mean(clip.hands_detected == 2):>9.1f}%"
                  f"{np.median(clip.inference_ms):>8.1f}ms{np.percentile(clip.inference_ms, 95):>8.1f}ms")
    all_counts = np.concatenate(counts)
    all_times = np.concatenate(times)
    return 100 * float(np.mean(all_counts == 2)), float(np.percentile(all_times, 95))


def report_session(session: str, clips: dict[str, Clip]) -> tuple[float, float]:
    """Calibra con los primeros segundos y evalúa con el resto. Devuelve (botón, cruceta)."""
    print(f"\n== Sesión '{session}' (calibración: primeros {CALIBRATION_FRAMES} frames válidos) ==")
    button = dpad = float("nan")

    if has_all(clips, BUTTON_LABELS):
        classifier = calibrate(clips, BUTTON_LABELS, "right", BUTTON_FEATURES)
        button = balanced_accuracy(classifier, clips, BUTTON_LABELS, "right", BUTTON_FEATURES,
                                   after_calibration=True)
        print(f"  Botón A   reposo vs pulsado: {button:5.1f} %")
        for label in ("boton_a_ritmo", "combo"):
            if label in clips:
                pressed = smooth_labels(classifier.predict(clips[label].features("right", BUTTON_FEATURES)),
                                        "boton_a_mantenido", SMOOTHING_FRAMES)
                seconds = len(pressed) / clips[label].fps
                print(f"            {label}: {count_presses(pressed)} pulsaciones en "
                      f"{seconds:.0f} s (esperado ~1 por segundo)")
    else:
        print("  Botón A: faltan clips 'reposo' o 'boton_a_mantenido'.")

    if has_all(clips, DPAD_LABELS):
        classifier = calibrate(clips, DPAD_LABELS, "left", DPAD_FEATURES)
        dpad = balanced_accuracy(classifier, clips, DPAD_LABELS, "left", DPAD_FEATURES,
                                 after_calibration=True)
        print(f"  Cruceta   reposo/izq/der:    {dpad:5.1f} %")
        if "combo" in clips:
            right_held = accuracy(classifier, clips["combo"].features("left", DPAD_FEATURES), "dpad_der")
            print(f"            combo leído como 'derecha': {right_held:5.1f} %  "
                  f"(pulsar A no debe mover el otro pulgar)")
    else:
        print("  Cruceta: faltan clips 'reposo', 'dpad_izq' o 'dpad_der'.")
    return button, dpad


def report_cross_session(sessions: dict[str, dict[str, Clip]]) -> float:
    """Calibrar en A y usar en B sin recalibrar. Devuelve el peor caso."""
    if len(sessions) < 2:
        print("\n== Entre sesiones ==\n  Solo hay una sesión: graba otra con --light distinto.")
        return float("nan")
    print("\n== Entre sesiones: calibrar en una y usar en otra SIN recalibrar ==")
    worst = float("inf")
    for source, target in permutations(sorted(sessions), 2):
        a, b = sessions[source], sessions[target]
        parts = []
        if has_all(a, BUTTON_LABELS) and has_all(b, BUTTON_LABELS):
            score = balanced_accuracy(calibrate(a, BUTTON_LABELS, "right", BUTTON_FEATURES), b,
                                      BUTTON_LABELS, "right", BUTTON_FEATURES, after_calibration=False)
            parts.append(f"botón {score:5.1f} %")
            worst = min(worst, score)
        if has_all(a, DPAD_LABELS) and has_all(b, DPAD_LABELS):
            score = balanced_accuracy(calibrate(a, DPAD_LABELS, "left", DPAD_FEATURES), b,
                                      DPAD_LABELS, "left", DPAD_FEATURES, after_calibration=False)
            parts.append(f"cruceta {score:5.1f} %")
            worst = min(worst, score)
        print(f"  {source:>10} → {target:<10} {'   '.join(parts)}")
    return worst


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #

def plot_feature_space(sessions: dict[str, dict[str, Clip]], hand: str,
                       labels: tuple[str, ...], filename: str, title: str) -> None:
    """Punta del pulgar respecto a la base del índice (dx, dy), por sesión y etiqueta."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "^", "s", "D"]
    for marker, (session, clips) in zip(markers, sorted(sessions.items())):
        for label in labels:
            if label not in clips:
                continue
            features = clips[label].left if hand == "left" else clips[label].right
            ax.scatter(features[:, 0], features[:, 1], s=4, alpha=0.35, marker=marker,
                       label=f"{session} · {label}")
    ax.invert_yaxis()  # en imagen, y crece hacia abajo
    ax.set_xlabel("dx (tamaños de mano)")
    ax.set_ylabel("dy (tamaños de mano)")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, dpi=120)
    plt.close(fig)


def plot_rhythm(sessions: dict[str, dict[str, Clip]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for session, clips in sorted(sessions.items()):
        if "boton_a_ritmo" in clips:
            clip = clips["boton_a_ritmo"]
            seconds = np.arange(len(clip.right)) / clip.fps
            ax.plot(seconds, clip.right[:, 1], linewidth=1, label=session)
    ax.set_title("boton_a_ritmo: altura de la punta del pulgar derecho (cada valle/pico = una pulsación)")
    ax.set_xlabel("segundos")
    ax.set_ylabel("dy (tamaños de mano)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "boton_a_ritmo.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Veredicto
# --------------------------------------------------------------------------- #

def verdict(two_hands: float, p95: float, button: float, dpad: float, cross: float) -> None:
    def line(ok: bool, text: str) -> None:
        print(f"{'OK ' if ok else 'NO '} {text}")

    print("\n================ VEREDICTO ================")
    line(two_hands >= TWO_HANDS_TARGET, f"2 manos detectadas          {two_hands:5.1f} %  (>= {TWO_HANDS_TARGET:.0f} %)")
    line(p95 <= INFERENCE_P95_TARGET_MS, f"inferencia p95              {p95:5.1f} ms (<= {INFERENCE_P95_TARGET_MS:.0f} ms)")
    line(button >= ACCURACY_TARGET, f"botón A, calibrado          {button:5.1f} %  (>= {ACCURACY_TARGET:.0f} %)")
    line(dpad >= ACCURACY_TARGET, f"cruceta, calibrada          {dpad:5.1f} %  (>= {ACCURACY_TARGET:.0f} %)")
    if np.isnan(cross):
        print("--  entre sesiones            sin datos (falta una segunda sesión)")
    else:
        line(cross >= ACCURACY_TARGET, f"entre sesiones, sin recal.  {cross:5.1f} %  (si NO: calibrar al arrancar)")


def main() -> None:
    sessions = load_clips()
    if not sessions:
        print(f"No hay landmarks en {LANDMARKS_DIR}. Ejecuta antes scripts/extract.py")
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    two_hands, p95 = report_detection(sessions)
    results = [report_session(session, clips) for session, clips in sorted(sessions.items())]
    cross = report_cross_session(sessions)

    plot_feature_space(sessions, "right", BUTTON_LABELS, "boton_a_espacio.png",
                       "Botón A: punta del pulgar derecho")
    plot_feature_space(sessions, "left", DPAD_LABELS, "cruceta_espacio.png",
                       "Cruceta: punta del pulgar izquierdo")
    plot_rhythm(sessions)

    worst_button = float(np.nanmin([r[0] for r in results]))
    worst_dpad = float(np.nanmin([r[1] for r in results]))
    verdict(two_hands, p95, worst_button, worst_dpad, cross)
    print(f"\nGráficas en {REPORT_DIR}")


if __name__ == "__main__":
    main()
