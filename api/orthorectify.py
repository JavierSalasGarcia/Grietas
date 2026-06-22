"""
Stub de ortorectificación — implementación completa en Fase 6.
Recibe imagen oblicua + metadata sidecar y devuelve (ortho, gsd_mm_px).
"""
import numpy as np


def orthorectify(image: np.ndarray, metadata: dict) -> tuple[np.ndarray, float | None]:
    """
    Transforma la foto oblicua en vista cenital métrica.

    Args:
        image:    imagen BGR (OpenCV) de la foto ciudadana.
        metadata: sidecar JSON enviado por la PWA.

    Returns:
        (ortho_image, gsd_mm_px) — la ortoiamgen y la resolución en mm/px.
        Si el pitch está fuera del rango aceptable, devuelve (image, None).
    """
    # Stub: devuelve la imagen sin transformar hasta que se implemente Fase 6.
    return image, None
