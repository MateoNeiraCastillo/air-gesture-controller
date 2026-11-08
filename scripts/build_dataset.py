"""Builds the training dataset from every guided session in data/guided/.

Each frame gets a label per hand from what the script asked for (see
air_controller/dataset.py). Older sessions, recorded before sessions saved the
person's nickname, are all Mateo's.

Usage:
    python scripts/build_dataset.py
Output:
    data/dataset/frames.npz   one row per frame: person, session, landmarks, labels
"""

from pathlib import Path

import numpy as np

from air_controller.dataset import (
    LEFT_CLASSES,
    RIGHT_CLASSES,
    UNKNOWN,
    label_counts,
    session_frames,
)
from air_controller.recordings import guided_sessions

ROOT = Path(__file__).resolve().parent.parent
GUIDED_DIR = ROOT / "data" / "guided"
OUT = ROOT / "data" / "dataset" / "frames.npz"
LEGACY_PERSON = "mateo"


def main() -> None:
    sessions = guided_sessions(GUIDED_DIR)
    if not sessions:
        print(f"No guided sessions in {GUIDED_DIR}. Record some with scripts/guided_session.py --script data.")
        return
    parts: dict[str, list[np.ndarray]] = {}
    print(f"{'session':<36}{'person':<10}{'script':<9}{'frames':>7}{'labelled L/R':>15}")
    for path in sessions:
        with np.load(path) as data:
            data = {key: data[key] for key in data.files}
        frames = session_frames(data)
        person = str(data["person"]) if "person" in data and str(data["person"]) else LEGACY_PERSON
        script = str(data["script_name"]) if "script_name" in data else "classic"
        n = len(frames["timestamp_ms"])
        frames["person"] = np.array([person] * n)
        frames["session"] = np.array([path.stem] * n)
        frames["script"] = np.array([script] * n)
        for key, values in frames.items():
            parts.setdefault(key, []).append(values)
        labelled = (f"{int(np.sum(frames['left_label'] != UNKNOWN))}/"
                    f"{int(np.sum(frames['right_label'] != UNKNOWN))}")
        print(f"{path.name:<36}{person:<10}{script:<9}{n:>7}{labelled:>15}")

    dataset = {key: np.concatenate(values) for key, values in parts.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **dataset)
    people = sorted(set(dataset["person"]))
    print(f"\n{len(dataset['timestamp_ms'])} frames from {len(sessions)} sessions and "
          f"{len(people)} people ({', '.join(people)})")
    print("left hand: ", label_counts(dataset["left_label"], LEFT_CLASSES))
    print("right hand:", label_counts(dataset["right_label"], RIGHT_CLASSES))
    print(f"Saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
