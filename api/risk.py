"""
Cálculo del índice de riesgo R ∈ [0, 10].

R(x,y,t) = 0.40 · severity_score_norm
          + 0.30 · |velocity_norm|
          + 0.20 · max(acceleration_norm, 0)
          + 0.10 · density_norm
"""

# Pesos del índice de riesgo (CONTEXT.md §7)
RISK_ALPHA = 0.40
RISK_BETA = 0.30
RISK_GAMMA = 0.20
RISK_DELTA = 0.10

# Valores de normalización (máximos esperados por componente)
_MAX_SEVERITY_SCORE = 10.0
_MAX_VELOCITY_MM_YR = 50.0   # mm/año — umbral de saturación
_MAX_ACCEL = 15.0             # mm/año²
_MAX_DENSITY = 1.0            # ya normalizado [0,1]


def _norm(value: float | None, max_val: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, abs(value) / max_val))


def compute_risk_index(report) -> float:
    """
    Calcula el índice de riesgo R ∈ [0, 10] para un Report SQLAlchemy.
    Componentes que aún no se han calculado (None) contribuyen con 0.
    """
    # Severidad de grieta
    severity_score = _severity_score_from_class(report.severity_class)
    s_norm = _norm(severity_score, _MAX_SEVERITY_SCORE)

    # Velocidad de subsidencia InSAR
    v_norm = _norm(report.subsid_vel_mm_yr, _MAX_VELOCITY_MM_YR)

    # Aceleración de subsidencia
    a_norm = _norm(
        max(report.subsid_accel, 0.0) if report.subsid_accel is not None else None,
        _MAX_ACCEL,
    )

    # Densidad de reportes (placeholder — se implementa en Fase 8)
    d_norm = 0.0

    risk = (
        RISK_ALPHA * s_norm
        + RISK_BETA * v_norm
        + RISK_GAMMA * a_norm
        + RISK_DELTA * d_norm
    ) * 10.0

    return round(min(10.0, max(0.0, risk)), 4)


def _severity_score_from_class(severity_class: str | None) -> float | None:
    """Convierte la clase textual de severidad a un score numérico."""
    mapping = {"leve": 2.5, "moderado": 6.0, "severo": 9.5}
    if severity_class is None:
        return None
    return mapping.get(severity_class.lower())
