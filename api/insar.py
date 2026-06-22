"""
Stub de consulta InSAR — implementación completa en Fase 8.
"""
from sqlalchemy.ext.asyncio import AsyncSession


async def get_subsidence_at_point(
    lng: float, lat: float, db: AsyncSession
) -> dict | None:
    """
    Retorna datos de subsidencia Sentinel-1 para el punto (lng, lat).

    Returns:
        Dict con keys: displacement_mm (list), velocity_mm_yr, acceleration.
        None si no hay datos InSAR disponibles en ese punto.
    """
    # Stub: retorna None hasta que se implemente Fase 8.
    return None
