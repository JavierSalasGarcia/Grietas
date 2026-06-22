"""
Stub de análisis de grieta — implementación completa en Fase 7.
Recibe la ortofoto y el GSD, devuelve métricas del vector v_grieta.
"""
import numpy as np


def analyze_crack(ortho: np.ndarray, gsd_mm_px: float | None) -> dict:
    """
    Segmenta la grieta y extrae métricas geométricas.

    Args:
        ortho:      ortofoto BGR (OpenCV).
        gsd_mm_px:  resolución del suelo en mm/px (None si no disponible).

    Returns:
        Diccionario con los campos de v_grieta más gsd_mm_px.
    """
    # Stub: retorna métricas nulas hasta que se implemente Fase 7.
    return {
        "crack_width_mm": None,
        "crack_length_mm": None,
        "crack_area_mm2": None,
        "crack_fractal_dim": None,
        "crack_angle_deg": None,
        "crack_area_ratio": None,
        "crack_branches": None,
        "severity_class": None,
        "severity_score": None,
        "gsd_mm_px": gsd_mm_px,
    }
