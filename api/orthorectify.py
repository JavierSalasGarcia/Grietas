"""
api/orthorectify.py
===================
Ortorectificación de fotografías oblicuas a vista cenital métrica.

Entrada:
  image    : np.ndarray BGR (de cv2.imread)
  metadata : dict (JSON sidecar de la PWA, campo 'camera' y 'citizen')

Salida:
  (ortho: np.ndarray, gsd_mm_px: float, quality: str)
  quality: "good" | "poor"  (poor si pitch fuera de 30°–60°)
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

PITCH_MIN_DEG = 30.0
PITCH_MAX_DEG = 60.0
MAX_OUTPUT_PX = 2048
DEFAULT_FOV_DEG = 70.0


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA MATRIZ INTRÍNSECA K
# ──────────────────────────────────────────────────────────────────────────────

def build_K(fov_h_deg: float, width_px: int, height_px: int) -> np.ndarray:
    """
    Construye la matriz intrínseca de la cámara desde el FoV horizontal.

    Asume píxeles cuadrados y punto principal centrado (válido para smartphones;
    error < 2% con distorsión moderada).
    """
    fov_rad = np.radians(fov_h_deg)
    fx = (width_px  / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = width_px  / 2.0
    cy = height_px / 2.0
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1 ]], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA MATRIZ DE ROTACIÓN R DESDE EL GIROSCOPIO
# ──────────────────────────────────────────────────────────────────────────────

def build_R(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """
    Convierte los ángulos de DeviceOrientationEvent a matriz de rotación 3×3.

    Convención del navegador:
      alpha: yaw (Z), beta: pitch (X), gamma: roll (Y)
    Aplicamos: R = Rz(alpha) · Rx(beta) · Ry(gamma)
    """
    R = Rotation.from_euler(
        "zxy",
        [alpha_deg, beta_deg, gamma_deg],
        degrees=True,
    )
    return R.as_matrix()


# ──────────────────────────────────────────────────────────────────────────────
# CÁLCULO DEL GSD (GROUND SAMPLING DISTANCE)
# ──────────────────────────────────────────────────────────────────────────────

def compute_gsd(h_m: float, fov_h_deg: float,
                pitch_deg: float, width_px: int) -> float:
    """
    Ground Sampling Distance: milímetros de terreno por píxel en la ortofoto.

    distancia_oblicua = h / cos(θ)
    cobertura_terreno = 2 × d × tan(FoV_h / 2)
    GSD = cobertura × 1000 / ancho_px
    """
    pitch_rad  = np.radians(np.clip(abs(pitch_deg), 1.0, 89.0))
    slant_m    = h_m / np.cos(pitch_rad)
    fov_rad    = np.radians(fov_h_deg)
    ground_w_m = 2.0 * slant_m * np.tan(fov_rad / 2.0)
    return (ground_w_m * 1000.0) / width_px


# ──────────────────────────────────────────────────────────────────────────────
# ESTIMACIÓN DE ALTURA DE CÁMARA
# ──────────────────────────────────────────────────────────────────────────────

def estimate_camera_height(metadata: dict) -> float:
    """
    Estima la altura de la cámara sobre el suelo en metros.

    Usa aproximación antropométrica: h ≈ altura_ciudadano × 0.85
    (teléfono sostenido a altura de hombro/cabeza).
    """
    height_cm = metadata.get("citizen", {}).get("height_cm", 165)
    return float(height_cm) / 100.0 * 0.85


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE ORTORECTIFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def orthorectify(
    image:    np.ndarray,
    metadata: dict,
) -> tuple[np.ndarray, float, str]:
    """
    Transforma la imagen oblicua a vista cenital métrica.

    Parámetros
    ----------
    image    : imagen BGR de OpenCV (tal como llega del upload)
    metadata : dict parseado del JSON sidecar de la PWA

    Retorna
    -------
    ortho    : np.ndarray — imagen ortorectificada (BGR)
    gsd      : float — milímetros por píxel en la ortofoto
    quality  : str  — "good" | "poor"
    """
    cam    = metadata.get("camera", {})
    orient = cam.get("orientation", {})
    res    = cam.get("resolution", {})

    W = int(res.get("width",  image.shape[1]))
    H = int(res.get("height", image.shape[0]))

    pitch_deg = float(orient.get("beta",  45.0))
    roll_deg  = float(orient.get("gamma",  0.0))
    yaw_deg   = float(orient.get("alpha",  0.0))
    fov_h_deg = float(cam.get("fov_estimate_deg", DEFAULT_FOV_DEG))

    quality = "good" if PITCH_MIN_DEG <= abs(pitch_deg) <= PITCH_MAX_DEG else "poor"
    if quality == "poor":
        h_m = estimate_camera_height(metadata)
        gsd = compute_gsd(h_m, fov_h_deg, 45.0, W)
        return image.copy(), gsd, quality

    K = build_K(fov_h_deg, W, H)
    R = build_R(yaw_deg, pitch_deg, roll_deg)
    h_m = estimate_camera_height(metadata)

    # Homografía plano suelo (z=0) → imagen: H = K [r1 | r2 | t]
    t = np.array([0.0, 0.0, -h_m])
    H_mat = K @ np.column_stack([R[:, 0], R[:, 1], t])

    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        gsd = compute_gsd(h_m, fov_h_deg, pitch_deg, W)
        return image.copy(), gsd, "poor"

    gsd = compute_gsd(h_m, fov_h_deg, pitch_deg, W)

    ground_w_mm = W * gsd
    ground_h_mm = H * gsd
    out_w = min(int(ground_w_mm), MAX_OUTPUT_PX)
    out_h = min(int(ground_h_mm), MAX_OUTPUT_PX)

    ortho = cv2.warpPerspective(
        image, H_inv, (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return ortho, gsd, quality


# ──────────────────────────────────────────────────────────────────────────────
# CORRECCIÓN DE INCERTIDUMBRE DEL GSD
# ──────────────────────────────────────────────────────────────────────────────

def gsd_uncertainty(gsd: float, intrinsics_source: str) -> tuple[float, float]:
    """
    Retorna (gsd_min, gsd_max) según la fuente de intrínsecos.
    Usado para reportar el rango métrico en lugar de un valor puntual.
    """
    error_factors = {
        "webxr":                  0.02,
        "onboarding_calibration": 0.05,
        "ua_database":            0.12,
        "prior":                  0.18,
    }
    factor = error_factors.get(intrinsics_source, 0.18)
    return gsd * (1 - factor), gsd * (1 + factor)
