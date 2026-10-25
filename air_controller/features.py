"""Features geométricas del pulgar, calculadas sobre landmarks en píxeles.

Las coordenadas deben venir en píxeles (x*ancho, y*alto), no normalizadas:
MediaPipe normaliza x e y por separado, así que en una imagen 640x480 los
ángulos calculados sobre coordenadas normalizadas salen deformados.
"""

import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9


def joint_angle_deg(a: np.ndarray, joint: np.ndarray, b: np.ndarray) -> float:
    """Ángulo en `joint` formado por los segmentos joint→a y joint→b, en grados."""
    u = a - joint
    v = b - joint
    norm_product = np.linalg.norm(u) * np.linalg.norm(v)
    if norm_product == 0:
        return float("nan")
    cosine = np.clip(np.dot(u, v) / norm_product, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def thumb_ip_angle(hand: np.ndarray) -> float:
    """Flexión en la articulación IP del pulgar (landmarks 2-3-4). 180° = recto."""
    return joint_angle_deg(hand[THUMB_MCP], hand[THUMB_IP], hand[THUMB_TIP])


def thumb_mcp_angle(hand: np.ndarray) -> float:
    """Flexión en la articulación MCP del pulgar (landmarks 1-2-3). Plan B si la IP no separa."""
    return joint_angle_deg(hand[THUMB_CMC], hand[THUMB_MCP], hand[THUMB_IP])


def hand_size(hand: np.ndarray) -> float:
    """Distancia muñeca → base del dedo medio: escala de referencia de la mano."""
    return float(np.linalg.norm(hand[MIDDLE_MCP, :2] - hand[WRIST, :2]))


def thumb_position_features(hand: np.ndarray) -> np.ndarray:
    """Posición de la punta del pulgar en "tamaños de mano", invariante a la distancia.

    Devuelve (dx, dy) respecto a la base del índice y (dx, dy) respecto a la muñeca.
    Es la feature que funcionó en el spike: los ángulos fallan cuando el pulgar
    se dobla sobre el puño, la posición no.
    """
    size = hand_size(hand)
    if size == 0 or np.isnan(hand).any():
        return np.full(4, np.nan)
    tip = hand[THUMB_TIP, :2]
    from_index = (tip - hand[INDEX_MCP, :2]) / size
    from_wrist = (tip - hand[WRIST, :2]) / size
    return np.concatenate([from_index, from_wrist])


def thumb_offset(hand: np.ndarray) -> np.ndarray:
    """Posición de la punta del pulgar respecto a la base del índice,
    en "tamaños de mano" (invariante a la distancia a la cámara). Devuelve (dx, dy)."""
    size = hand_size(hand)
    if size == 0:
        return np.array([np.nan, np.nan])
    return (hand[THUMB_TIP, :2] - hand[INDEX_MCP, :2]) / size
