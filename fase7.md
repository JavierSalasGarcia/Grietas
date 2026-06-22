# Fase 7 — Segmentación de Grieta e Índice de Severidad `[PENDIENTE]`

## Contexto del proyecto

Sistema de reporte de grietas en calles de Toluca. Esta fase recibe la ortofoto
métrica de Fase 6 y produce:
1. Una máscara binaria de la grieta (segmentación pixel-wise)
2. El vector de severidad `v_grieta` con todas las medidas en milímetros reales
3. La clasificación de severidad según ASTM D6433 simplificado

---

## Prerrequisitos

- Fase 6 completada: `orthorectify()` produce `(ortho: np.ndarray, gsd: float, quality: str)`
- La ortofoto tiene dimensiones ≤ 2048×2048 px con GSD conocido en mm/px
- `pip install opencv-python-headless numpy scikit-image scipy`
- Pesos de DeepCrack descargados (ver instrucciones abajo)

---

## Modelo de segmentación: DeepCrack

**DeepCrack** (Liu et al., 2019) es una CNN jerárquica entrenada específicamente
para segmentar grietas en pavimentos. Supera a U-Net genérico en grietas finas
porque usa múltiples escalas de supervisión durante el entrenamiento.

**Referencia**: Liu, Y. et al. (2019). *DeepCrack: A Deep Hierarchical Feature
Learning Architecture for Crack Segmentation*. Neurocomputing, 338, 139–153.

**Repositorio oficial**: `https://github.com/yhlleo/DeepCrack`

### Descargar pesos preentrenados

```bash
# Opción A: clonar repositorio y descargar pesos
git clone https://github.com/yhlleo/DeepCrack.git external/deepcrack
# Descargar pesos desde el README del repo (enlace de Google Drive o Baidu)
# Colocar en: external/deepcrack/checkpoints/deepcrack_pretrained.pth

# Opción B (alternativa si DeepCrack no está disponible):
# Usar SegFormer-B2 preentrenado en CrackSeg9k (HuggingFace Hub)
# pip install transformers
# from transformers import SegformerForSemanticSegmentation
# model = SegformerForSemanticSegmentation.from_pretrained("nickmuchi/segformer-b2-finetuned-segments-crackSeg9k")
```

---

## Archivo a crear

```
api/crack_analysis.py
```

---

## Implementación completa

```python
"""
api/crack_analysis.py
=====================
Segmentación de grietas y extracción del vector de severidad métrico.

Entrada:
  ortho     : np.ndarray BGR (ortofoto de Fase 6)
  gsd_mm_px : float (milímetros por píxel)

Salida:
  dict con métricas en mm y mm²
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import label as scipy_label
from skimage.morphology import skeletonize, thin
from skimage.measure import regionprops


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

DEEPCRACK_WEIGHTS = Path("external/deepcrack/checkpoints/deepcrack_pretrained.pth")
DEEPCRACK_INPUT_SIZE = (512, 512)   # tamaño de entrada del modelo


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO DE SEGMENTACIÓN
# ──────────────────────────────────────────────────────────────────────────────

_segmentation_model = None

def _load_model():
    """
    Carga DeepCrack (o fallback) una sola vez.
    Si los pesos no están disponibles, usa un segmentador clásico como fallback.
    """
    global _segmentation_model
    if _segmentation_model is not None:
        return _segmentation_model

    if DEEPCRACK_WEIGHTS.exists():
        import sys
        sys.path.insert(0, "external/deepcrack")
        import torch
        from models.deepcrack import DeepCrack   # del repo clonado
        model = DeepCrack()
        state = torch.load(DEEPCRACK_WEIGHTS, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _segmentation_model = ("deepcrack", model)
    else:
        # Fallback: segmentación clásica (Canny + morfología)
        _segmentation_model = ("classic", None)

    return _segmentation_model


# ──────────────────────────────────────────────────────────────────────────────
# SEGMENTACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def segment_crack(ortho_bgr: np.ndarray) -> np.ndarray:
    """
    Produce una máscara binaria de la grieta.
    Retorna array uint8 (0 = fondo, 255 = grieta), mismo tamaño que ortho_bgr.
    """
    model_type, model = _load_model()

    if model_type == "deepcrack":
        return _segment_deepcrack(ortho_bgr, model)
    else:
        return _segment_classic(ortho_bgr)


def _segment_deepcrack(ortho_bgr: np.ndarray, model) -> np.ndarray:
    """Inferencia con DeepCrack usando PyTorch."""
    import torch

    H_orig, W_orig = ortho_bgr.shape[:2]
    img_resized = cv2.resize(ortho_bgr, DEEPCRACK_INPUT_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor      = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    tensor      = tensor.unsqueeze(0)   # (1, 3, 512, 512)

    with torch.no_grad():
        output = model(tensor)
        # DeepCrack devuelve múltiples escalas; usar la última (más detallada)
        pred = torch.sigmoid(output[-1]).squeeze().numpy()

    # Umbralizar y redimensionar al tamaño original
    mask_small = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (W_orig, H_orig),
                      interpolation=cv2.INTER_NEAREST)
    return mask


def _segment_classic(ortho_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback cuando DeepCrack no está disponible.
    Usa detección de bordes adaptativa + morfología.
    Funciona razonablemente para grietas con buen contraste.
    """
    gray  = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE para mejorar contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    # Umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    # Morfología para limpiar ruido y conectar la grieta
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS MORFOLÓGICO Y EXTRACCIÓN DE MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────────

def box_counting_dimension(binary_mask: np.ndarray, n_levels: int = 8) -> float:
    """
    Dimensión fractal por box-counting.
    D_f ∈ [1.0, 2.0] para grietas en 2D:
      1.0 = grieta perfectamente recta
      1.5 = grieta ramificada moderada
      >1.7 = red de grietas muy compleja
    """
    sizes  = [2 ** i for i in range(1, n_levels + 1)]
    counts = []
    img    = binary_mask.astype(bool)
    for size in sizes:
        n_rows = int(np.ceil(img.shape[0] / size))
        n_cols = int(np.ceil(img.shape[1] / size))
        count  = 0
        for r in range(n_rows):
            for c in range(n_cols):
                block = img[r*size:(r+1)*size, c*size:(c+1)*size]
                if block.any():
                    count += 1
        counts.append(count)

    # Ajuste log-log: pendiente = dimensión fractal
    log_sizes  = np.log(1.0 / np.array(sizes, dtype=float))
    log_counts = np.log(np.array(counts, dtype=float) + 1e-9)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(np.clip(coeffs[0], 1.0, 2.0))


def dominant_orientation(binary_mask: np.ndarray) -> float:
    """
    Orientación dominante de la grieta en grados [0°, 180°).
    0° = horizontal, 90° = vertical.
    Usa momentos de Hu del esqueleto para calcular el eje principal.
    """
    skeleton = skeletonize(binary_mask > 0).astype(np.uint8)
    if skeleton.sum() < 10:
        return 0.0

    points = np.column_stack(np.where(skeleton > 0)).astype(np.float64)
    # PCA manual: el primer componente es el eje dominante
    cov    = np.cov(points.T)
    _, vecs = np.linalg.eigh(cov)
    # Vector del eje principal (mayor eigenvector)
    axis   = vecs[:, -1]
    angle  = np.degrees(np.arctan2(axis[0], axis[1])) % 180.0
    return float(angle)


def count_branch_points(skeleton: np.ndarray) -> int:
    """
    Cuenta los puntos de bifurcación del esqueleto de la grieta.
    Un punto es bifurcación si tiene ≥ 3 vecinos en los 8 alrededores.
    """
    kernel    = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) - skeleton
    branch_pts = (skeleton > 0) & (neighbors >= 3)
    return int(branch_pts.sum())


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def analyze_crack(ortho_bgr: np.ndarray, gsd_mm_px: float) -> dict:
    """
    Pipeline completo: segmentación → morfología → métricas métricas.

    Parámetros
    ----------
    ortho_bgr : np.ndarray — imagen BGR ortorectificada (de Fase 6)
    gsd_mm_px : float      — milímetros por píxel (de Fase 6)

    Retorna
    -------
    dict con:
      crack_width_mm    : float — ancho medio de la grieta en mm
      crack_length_mm   : float — longitud total del esqueleto en mm
      crack_area_mm2    : float — área total de píxeles de grieta en mm²
      crack_fractal_dim : float — dimensión fractal [1.0–2.0]
      crack_angle_deg   : float — orientación dominante [0°–180°)
      crack_area_ratio  : float — fracción de área dañada [0–1]
      crack_branches    : int   — número de bifurcaciones
      severity_class    : str   — "leve" | "moderado" | "severo"
      severity_score    : float — puntuación numérica [0–10]
      gsd_mm_px         : float — GSD usado en el cálculo
      mask_base64       : str   — máscara PNG en base64 (para visualización)
    """
    # 1. Segmentación
    mask = segment_crack(ortho_bgr)   # uint8: 0 o 255

    n_crack_px = int((mask > 0).sum())
    if n_crack_px < 10:
        # No se detectó grieta significativa
        return _empty_metrics(gsd_mm_px)

    # 2. Esqueleto (línea central de la grieta)
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    n_skel_px = int(skeleton.sum())

    # 3. Métricas métricas (convertir px → mm)
    L_mm   = n_skel_px  * gsd_mm_px
    A_mm2  = n_crack_px * (gsd_mm_px ** 2)
    w_mean = A_mm2 / L_mm if L_mm > 0 else 0.0

    # 4. Métricas geométricas
    D_f        = box_counting_dimension(mask)
    theta_deg  = dominant_orientation(mask)
    n_branches = count_branch_points(skeleton)
    A_ratio    = n_crack_px / (ortho_bgr.shape[0] * ortho_bgr.shape[1])

    # 5. Clasificación de severidad (ASTM D6433 simplificado)
    sev_class, sev_score = _classify_severity(w_mean, A_ratio)

    # 6. Codificar máscara en base64 para almacenar en DB / enviar al dashboard
    import base64
    _, buf = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(buf).decode("ascii")

    return {
        "crack_width_mm":    round(w_mean, 2),
        "crack_length_mm":   round(L_mm, 1),
        "crack_area_mm2":    round(A_mm2, 1),
        "crack_fractal_dim": round(D_f, 3),
        "crack_angle_deg":   round(theta_deg, 1),
        "crack_area_ratio":  round(A_ratio, 5),
        "crack_branches":    n_branches,
        "severity_class":    sev_class,
        "severity_score":    sev_score,
        "gsd_mm_px":         round(gsd_mm_px, 3),
        "mask_base64":       mask_b64,
    }


def _classify_severity(w_mm: float, a_ratio: float) -> tuple[str, float]:
    """
    Clasificación de severidad y puntuación numérica.

    Criterios (basados en ASTM D6433 y norma INVIAS de Colombia):
      leve     : w < 0.3 mm  o área < 0.5%
      moderado : 0.3 ≤ w < 3.0 mm  y área 0.5%–5%
      severo   : w ≥ 3.0 mm  o área > 5%
    """
    if w_mm < 0.3 or a_ratio < 0.005:
        return "leve", 2.0
    elif w_mm < 3.0 and a_ratio < 0.05:
        # Escala continua dentro de "moderado" [3.0–6.0]
        score = 3.0 + (w_mm / 3.0) * 2.0 + (a_ratio / 0.05)
        return "moderado", round(min(score, 6.0), 1)
    else:
        # Escala continua dentro de "severo" [6.0–10.0]
        score = 6.0 + min(w_mm / 10.0 * 2.0 + a_ratio * 10, 4.0)
        return "severo", round(min(score, 10.0), 1)


def _empty_metrics(gsd_mm_px: float) -> dict:
    return {
        "crack_width_mm": 0.0, "crack_length_mm": 0.0, "crack_area_mm2": 0.0,
        "crack_fractal_dim": 1.0, "crack_angle_deg": 0.0, "crack_area_ratio": 0.0,
        "crack_branches": 0, "severity_class": "leve", "severity_score": 0.0,
        "gsd_mm_px": gsd_mm_px, "mask_base64": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# EVOLUCIÓN TEMPORAL
# ──────────────────────────────────────────────────────────────────────────────

def compute_temporal_evolution(
    current:    dict,
    previous:   dict,
    delta_days: float,
) -> dict:
    """
    Calcula el cambio en las métricas entre dos reportes del mismo punto.

    Parámetros
    ----------
    current    : métricas del reporte actual (de analyze_crack)
    previous   : métricas del reporte anterior (de la DB)
    delta_days : días entre ambos reportes

    Retorna
    -------
    dict con deltas y tasas de crecimiento
    """
    delta_w = current["crack_width_mm"]  - previous["crack_width_mm"]
    delta_L = current["crack_length_mm"] - previous["crack_length_mm"]
    delta_A = current["crack_area_mm2"]  - previous["crack_area_mm2"]

    months  = max(delta_days / 30.0, 0.001)   # evitar división por cero

    return {
        "delta_width_mm":       round(delta_w, 2),
        "delta_length_mm":      round(delta_L, 1),
        "delta_area_mm2":       round(delta_A, 1),
        "growth_rate_mm_month": round(delta_L / months, 2),
        "widening_rate_mm_month": round(delta_w / months, 3),
        "delta_days":           delta_days,
        "deteriorating":        delta_w > 0.5 or delta_L > 10.0,
    }
```

---

## Prueba unitaria del módulo

```python
# tests/test_crack_analysis.py
import numpy as np
import pytest
from api.crack_analysis import (
    box_counting_dimension, dominant_orientation,
    count_branch_points, _classify_severity, _empty_metrics
)

def make_crack_mask(size=256, crack_width=5):
    """Crea una máscara sintética con una grieta horizontal."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mid  = size // 2
    mask[mid - crack_width//2 : mid + crack_width//2, :] = 255
    return mask

def test_classify_severity_leve():
    cls, score = _classify_severity(w_mm=0.1, a_ratio=0.001)
    assert cls == "leve"
    assert score == pytest.approx(2.0)

def test_classify_severity_moderado():
    cls, score = _classify_severity(w_mm=1.5, a_ratio=0.02)
    assert cls == "moderado"
    assert 3.0 <= score <= 6.0

def test_classify_severity_severo():
    cls, score = _classify_severity(w_mm=5.0, a_ratio=0.10)
    assert cls == "severo"
    assert score > 6.0

def test_box_counting_line():
    """Una línea recta tiene D_f ≈ 1.0."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[128, :] = 255
    D_f = box_counting_dimension(mask, n_levels=6)
    assert 0.9 <= D_f <= 1.3, f"D_f de línea recta inesperado: {D_f}"

def test_dominant_orientation_horizontal():
    mask = make_crack_mask()
    angle = dominant_orientation(mask)
    # Grieta horizontal → ángulo cerca de 0° o 180°
    assert angle < 30 or angle > 150, f"Ángulo inesperado: {angle}"

def test_empty_metrics_on_no_crack():
    result = _empty_metrics(gsd_mm_px=0.7)
    assert result["crack_width_mm"] == 0.0
    assert result["severity_class"] == "leve"
```

---

## Vector de severidad completo

El `analyze_crack()` produce el vector que se almacena en la tabla `reports` y
se usa en el índice de riesgo de Fase 8:

```
v_grieta = {
  crack_width_mm:    float   # ancho medio [mm]  — indicador primario
  crack_length_mm:   float   # longitud total [mm]
  crack_area_mm2:    float   # área [mm²]
  crack_fractal_dim: float   # complejidad topológica [1.0–2.0]
  crack_angle_deg:   float   # orientación respecto a imagen [0°–180°]
  crack_area_ratio:  float   # fracción dañada [0–1]
  crack_branches:    int     # número de bifurcaciones
  severity_class:    str     # "leve" | "moderado" | "severo"
  severity_score:    float   # puntuación continua [0–10]
  gsd_mm_px:         float   # resolución usada para calcular todo lo anterior
}
```

---

## Siguiente paso

**Fase 8** — Integración InSAR Sentinel-1 (`fase8.md`): ingestar los datos de
subsidencia y correlacionarlos con la ubicación GPS de cada reporte para
calcular el índice de riesgo integrado R(x,y,t).
