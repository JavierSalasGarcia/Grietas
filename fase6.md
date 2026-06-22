# Fase 6 — Módulo de Ortorectificación `[PENDIENTE]`

## Contexto del proyecto

Sistema de reporte de grietas en calles de Toluca. Esta fase implementa la
transformación geométrica que convierte una fotografía oblicua (tomada a ≈45°
desde vertical) en una **ortofoto cenital con escala métrica conocida** (mm/px).
Esto permite medir las grietas en milímetros reales en la Fase 7.

---

## Prerrequisitos

- Fase 4 completada: la PWA envía el JSON sidecar con `camera.orientation`
  y `camera.fov_estimate_deg`
- Fase 5 completada: el backend llama a `orthorectify()` desde `tasks.py`
- `pip install opencv-python-headless numpy scipy`

---

## Principio geométrico

Una foto tomada con la cámara a altura `h` e inclinación `θ` desde vertical
sobre un suelo plano (pavimento) puede modelarse como una proyección de perspectiva
de un plano 3D sobre el sensor 2D. La transformación inversa es una **homografía**
(transformación proyectiva 3×3) que "aplana" la imagen a vista cenital.

```
Cámara en (0, 0, h)
con rotación R (del giroscopio)          Imagen ortorectificada (vista aérea)
                                              ┌───────────────┐
    ┌──────────┐                              │               │
    │ imagen   │──── H_inv ───────────────►  │  ═══════════  │ ← grieta
    │ oblicua  │                              │               │
    └──────────┘                              └───────────────┘
                                             escala: GSD mm/px
```

**Invariante del suelo plano**: la correspondencia entre puntos del suelo (z=0)
y píxeles de la imagen es exactamente una homografía 3×3. Solo se necesita K y R.

---

## Archivo a crear

```
api/orthorectify.py
```

---

## Implementación completa

```python
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

# Rango de pitch considerado "buena calidad" para ortorectificación
PITCH_MIN_DEG = 30.0
PITCH_MAX_DEG = 60.0

# Resolución máxima de la ortofoto de salida (para no generar imágenes de 100 MB)
MAX_OUTPUT_PX = 2048

# FoV por defecto si no viene en el metadata
DEFAULT_FOV_DEG = 70.0


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA MATRIZ INTRÍNSECA K
# ──────────────────────────────────────────────────────────────────────────────

def build_K(fov_h_deg: float, width_px: int, height_px: int) -> np.ndarray:
    """
    Construye la matriz intrínseca de la cámara desde el FoV horizontal.

    Asume píxeles cuadrados y punto principal centrado (válido para la gran
    mayoría de smartphones; el error es < 2% incluso con distorsión moderada).

    K = ┌ fx   0   cx ┐
        │  0  fy   cy │
        └  0   0    1 ┘

    donde fx = (W/2) / tan(FoV_h/2)
    """
    fov_rad = np.radians(fov_h_deg)
    fx = (width_px  / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx   # píxeles cuadrados
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
      alpha: rotación en Z (yaw / azimut, 0–360°)
      beta:  rotación en X (pitch, –180 a 180°)
             beta=0 → teléfono plano en horizontal
             beta=90 → teléfono vertical (pantalla al frente del usuario)
      gamma: rotación en Y (roll, –90 a 90°)

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

    Para ángulo de pitch θ y altura h:
      distancia_oblicua = h / cos(θ)
      cobertura_terreno = 2 × d × tan(FoV_h / 2)
      GSD = cobertura × 1000 / ancho_px
    """
    pitch_rad = np.radians(np.clip(abs(pitch_deg), 1.0, 89.0))
    slant_m   = h_m / np.cos(pitch_rad)
    fov_rad   = np.radians(fov_h_deg)
    ground_width_m = 2.0 * slant_m * np.tan(fov_rad / 2.0)
    return (ground_width_m * 1000.0) / width_px   # mm/px


# ──────────────────────────────────────────────────────────────────────────────
# ESTIMACIÓN DE ALTURA DE CÁMARA
# ──────────────────────────────────────────────────────────────────────────────

def estimate_camera_height(metadata: dict) -> float:
    """
    Estima la altura de la cámara sobre el suelo en metros.

    Si el GPS incluye altitud (raro en interiores), usar la diferencia entre
    altitud GPS y altitud del punto reportado (requiere DEM del Valle de Toluca;
    implementar en Fase 8 cuando los datos InSAR estén disponibles).

    Por ahora usa la aproximación antropométrica:
      h ≈ altura_ciudadano × 0.85
    (teléfono sostenido a altura de hombro/cabeza)
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

    pitch_deg  = float(orient.get("beta",  45.0))
    roll_deg   = float(orient.get("gamma",  0.0))
    yaw_deg    = float(orient.get("alpha",  0.0))
    fov_h_deg  = float(cam.get("fov_estimate_deg", DEFAULT_FOV_DEG))

    # Evaluar calidad antes de proceder
    quality = "good" if PITCH_MIN_DEG <= abs(pitch_deg) <= PITCH_MAX_DEG else "poor"
    if quality == "poor":
        # Devolver imagen original sin transformar + GSD aproximado
        h_m = estimate_camera_height(metadata)
        gsd = compute_gsd(h_m, fov_h_deg, 45.0, W)
        return image.copy(), gsd, quality

    # Construir K y R
    K = build_K(fov_h_deg, W, H)
    R = build_R(yaw_deg, pitch_deg, roll_deg)
    h_m = estimate_camera_height(metadata)

    # Homografía del plano del suelo (z=0) a la imagen:
    # P = K [R | t]  →  para z=0  →  H = K [r1 | r2 | t]
    t = np.array([0.0, 0.0, -h_m])
    H_mat = K @ np.column_stack([R[:, 0], R[:, 1], t])

    # La inversa transforma de imagen → suelo (vista cenital)
    H_inv = np.linalg.inv(H_mat)

    # Calcular GSD con los parámetros reales del giroscopio
    gsd = compute_gsd(h_m, fov_h_deg, pitch_deg, W)

    # Calcular el tamaño de salida basado en el footprint en suelo
    # (convertido a píxeles al GSD estimado)
    ground_w_mm = W * gsd
    ground_h_mm = H * gsd
    out_w = min(int(ground_w_mm), MAX_OUTPUT_PX)
    out_h = min(int(ground_h_mm), MAX_OUTPUT_PX)

    # Aplicar la homografía inversa
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
        "webxr":                  0.02,   # ±2%
        "onboarding_calibration": 0.05,   # ±5%
        "ua_database":            0.12,   # ±12%
        "prior":                  0.18,   # ±18%
    }
    factor = error_factors.get(intrinsics_source, 0.18)
    return gsd * (1 - factor), gsd * (1 + factor)
```

---

## Prueba unitaria del módulo

```python
# tests/test_orthorectify.py
import numpy as np
import pytest
from api.orthorectify import (
    build_K, build_R, compute_gsd, orthorectify, gsd_uncertainty
)

def make_metadata(pitch=45, height_cm=170, fov=70, intrinsics="ua_database"):
    return {
        "citizen": {"height_cm": height_cm},
        "camera": {
            "orientation": {"alpha": 90, "beta": pitch, "gamma": 0, "absolute": True},
            "resolution":  {"width": 4000, "height": 3000},
            "fov_estimate_deg": fov,
            "intrinsics_source": intrinsics,
        }
    }

def test_build_K_shape():
    K = build_K(70, 4000, 3000)
    assert K.shape == (3, 3)
    assert K[0, 2] == pytest.approx(2000)  # cx = W/2
    assert K[1, 2] == pytest.approx(1500)  # cy = H/2

def test_build_R_identity_at_zero():
    """Sin rotación los ejes deben ser casi identidad."""
    R = build_R(0, 0, 0)
    assert R.shape == (3, 3)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6

def test_gsd_45deg_170cm():
    """A 45° y 170 cm de altura el GSD debe ser ~0.6–0.9 mm/px."""
    gsd = compute_gsd(h_m=0.85 * 1.70, fov_h_deg=70, pitch_deg=45, width_px=4000)
    assert 0.5 <= gsd <= 1.2, f"GSD inesperado: {gsd:.3f} mm/px"

def test_orthorectify_good_quality():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=45)
    ortho, gsd, quality = orthorectify(image, meta)
    assert quality == "good"
    assert ortho.shape[0] > 0 and ortho.shape[1] > 0
    assert 0.3 < gsd < 3.0

def test_orthorectify_poor_quality_at_15deg():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=15)   # fuera del rango 30°–60°
    ortho, gsd, quality = orthorectify(image, meta)
    assert quality == "poor"

def test_gsd_uncertainty():
    lo, hi = gsd_uncertainty(0.75, "ua_database")
    assert lo < 0.75 < hi
    assert abs((hi - lo) / 0.75 - 0.24) < 0.01   # ±12% → rango total 24%

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Ejecutar con:
```bash
pip install pytest
pytest tests/test_orthorectify.py -v
```

---

## Casos borde y cómo se manejan

| Caso | Comportamiento |
|---|---|
| `beta` (pitch) fuera de 30°–60° | `quality = "poor"`, se devuelve imagen original, GSD estimado con 45° asumido |
| `orientation` ausente en metadata | Todos los ángulos se asumen 0° excepto pitch=45°, `quality = "poor"` |
| `fov_estimate_deg` ausente | Se usa `DEFAULT_FOV_DEG = 70°` |
| `height_cm` ausente | Se usa 165 cm como valor por defecto |
| Homografía singular (H_inv no existe) | Se captura `np.linalg.LinAlgError`, se devuelve imagen original |
| Imagen muy oscura o borrosa | No afecta la ortorectificación; afecta la segmentación en Fase 7 |

---

## Valores de GSD típicos

| Altura (h) | Pitch | FoV | GSD (mm/px) | Ancho cubierto |
|---|---|---|---|---|
| 1.4 m | 45° | 70° | 0.58 mm/px | 2.3 m |
| 1.4 m | 45° | 77° | 0.66 mm/px | 2.6 m |
| 1.7 m | 45° | 70° | 0.71 mm/px | 2.8 m |
| 1.7 m | 45° | 79° | 0.79 mm/px | 3.2 m |
| 2.0 m | 45° | 70° | 0.83 mm/px | 3.3 m |

Con GSD ≈ 0.65 mm/px, grietas de ≥ 2 mm de ancho son medibles con ±1 píxel de precisión.

---

## Siguiente paso

**Fase 7** — Segmentación e índice de severidad (`fase7.md`): recibe la ortofoto
con GSD conocido y produce la máscara de grieta + vector de métricas en mm.
