# Fase 8 — Integración InSAR Sentinel-1 `[PENDIENTE]`

## Contexto del proyecto

Sistema de reporte de grietas en calles de Toluca. Esta fase ingesta los datos
de subsidencia del suelo obtenidos del satélite Sentinel-1 mediante interferometría
SAR (InSAR) y los combina con los reportes ciudadanos para calcular un **índice
de riesgo integrado R(x,y,t)** que prioriza zonas de mantenimiento vial.

---

## Prerrequisitos

- Fase 5 completada: API con PostgreSQL + PostGIS
- Fase 7 completada: tabla `reports` tiene columnas `crack_*` y `severity_score`
- Datos InSAR procesados disponibles en formato GeoTIFF (ver sección "Datos de entrada")
- `pip install rasterio geopandas shapely numpy scipy asyncpg sqlalchemy`

---

## Datos de entrada: características de los datos InSAR

| Parámetro | Valor |
|---|---|
| Satélite | Sentinel-1A/B (ESA, banda C, λ = 5.6 cm) |
| Modo | IW (Interferometric Wide Swath) |
| Resolución espacial | 15 m/píxel |
| Resolución temporal | 12 días por adquisición |
| Cobertura temporal | 2014 – presente (se actualiza con cada nueva imagen) |
| Cobertura espacial | Valle de Toluca (~60×60 km) |
| Procesamiento previo | PS-InSAR o SBAS con SNAP + StaMPS / MintPy |
| Producto | Desplazamiento acumulado en LOS (Line Of Sight) en mm |
| Coordenadas | WGS84 (EPSG:4326) |
| Formato | GeoTIFF, una banda, valores en mm (float32) |

**Nota sobre LOS vs. subsidencia vertical**: Sentinel-1 mide el desplazamiento
en la dirección LOS (inclinada ~38° desde vertical). Para el Valle de Toluca,
donde la subsidencia es predominantemente vertical (hundimiento por extracción
de agua), la conversión es: `subsidencia_vertical = LOS / cos(38°) ≈ LOS × 1.27`.

---

## Estructura de archivos a crear

```
api/
  insar.py           ← ingesta y consulta de datos InSAR
  risk.py            ← cálculo del índice de riesgo integrado

insar_data/          ← carpeta para los GeoTIFF (excluida del git)
  toluca_YYYYMMDD.tif    ← un archivo por adquisición
  toluca_velocity.tif    ← velocidad media del período completo
  metadata.json          ← {epsg, resolution_m, bbox, dates, n_files}
```

`insar_data/` debe estar en `.gitignore` (puede pesar decenas de GB).

---

## Esquema de la tabla de subsidencia en PostgreSQL

```sql
-- Ejecutar una vez, antes de ingestar los datos
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS insar_displacement (
    id              BIGSERIAL PRIMARY KEY,
    acquisition_date DATE NOT NULL,
    location         GEOMETRY(Point, 4326) NOT NULL,
    los_mm           FLOAT NOT NULL,      -- desplazamiento LOS en mm
    vert_mm          FLOAT,               -- convertido a vertical (LOS / cos 38°)
    pixel_x          INT,                 -- columna en el raster original
    pixel_y          INT                  -- fila en el raster original
);

-- Índice espacial para consultas rápidas por vecindad
CREATE INDEX IF NOT EXISTS idx_insar_location
    ON insar_displacement USING GIST (location);

-- Índice temporal
CREATE INDEX IF NOT EXISTS idx_insar_date
    ON insar_displacement (acquisition_date);

-- Vista materializada: velocidad por punto (mm/año)
-- (calcular tras ingestar al menos 6 meses de datos)
CREATE MATERIALIZED VIEW IF NOT EXISTS insar_velocity AS
SELECT
    pixel_x, pixel_y,
    AVG(ST_X(location::geometry)) AS lng,
    AVG(ST_Y(location::geometry)) AS lat,
    -- Regresión lineal temporal: pendiente = velocidad mm/año
    (regr_slope(vert_mm,
        EXTRACT(EPOCH FROM acquisition_date) / (365.25 * 86400)
    )) AS velocity_mm_yr,
    COUNT(*) AS n_acquisitions
FROM insar_displacement
GROUP BY pixel_x, pixel_y;

CREATE UNIQUE INDEX ON insar_velocity (pixel_x, pixel_y);
```

---

## api/insar.py

```python
"""
api/insar.py
============
Módulo de ingesta y consulta de datos InSAR de subsidencia.

Dos operaciones principales:
  1. ingest_insar_tif()       → Lee GeoTIFF y escribe en PostgreSQL
  2. get_subsidence_timeseries() → Consulta serie temporal para un punto GPS
"""

import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import rowcol
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger("insar")

# Factor de conversión LOS → vertical para Sentinel-1
# Ángulo de incidencia ≈ 38° para modo IW sobre el Valle de Toluca
LOS_TO_VERTICAL = 1.0 / math.cos(math.radians(38.0))   # ≈ 1.269


# ──────────────────────────────────────────────────────────────────────────────
# INGESTA
# ──────────────────────────────────────────────────────────────────────────────

async def ingest_insar_tif(
    tif_path:         str,
    acquisition_date: date,
    db:               AsyncSession,
    subsample:        int = 1,
) -> int:
    """
    Lee un GeoTIFF de desplazamiento LOS y lo inserta en la tabla
    insar_displacement. Retorna el número de píxeles insertados.

    subsample: tomar 1 de cada N píxeles para reducir volumen de datos.
               1 = todos los píxeles (~16 M para todo el valle a 15m).
               4 = 1 de cada 16 píxeles (~1 M, mucho más manejable).
               Usar subsample=4 para la primera ingesta; ajustar si se
               necesita mayor resolución espacial.

    Para datos en producción con millones de píxeles usar COPY en lugar
    de inserciones individuales (ver _bulk_insert).
    """
    with rasterio.open(tif_path) as src:
        data      = src.read(1)         # banda única (float32, mm LOS)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata

    rows, cols = data.shape
    records    = []

    for r in range(0, rows, subsample):
        for c in range(0, cols, subsample):
            val = float(data[r, c])
            if nodata is not None and abs(val - nodata) < 0.001:
                continue
            if not math.isfinite(val):
                continue

            # Convertir píxel → coordenadas geográficas
            lng, lat = rasterio.transform.xy(transform, r, c)

            records.append({
                "date":    acquisition_date,
                "lng":     lng,
                "lat":     lat,
                "los_mm":  round(val, 3),
                "vert_mm": round(val * LOS_TO_VERTICAL, 3),
                "px":      c,
                "py":      r,
            })

    if not records:
        log.warning("Ningún píxel válido en %s", tif_path)
        return 0

    await _bulk_insert(records, db)
    log.info("Ingestado %s: %d píxeles para %s", Path(tif_path).name,
             len(records), acquisition_date)
    return len(records)


async def _bulk_insert(records: list, db: AsyncSession):
    """Inserta en lotes de 10 000 usando COPY para máxima eficiencia."""
    BATCH = 10_000
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        values = ",".join(
            f"('{r['date']}', ST_SetSRID(ST_Point({r['lng']},{r['lat']}),4326), "
            f"{r['los_mm']}, {r['vert_mm']}, {r['px']}, {r['py']})"
            for r in batch
        )
        await db.execute(text(f"""
            INSERT INTO insar_displacement
              (acquisition_date, location, los_mm, vert_mm, pixel_x, pixel_y)
            VALUES {values}
            ON CONFLICT DO NOTHING
        """))
    await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# CONSULTA
# ──────────────────────────────────────────────────────────────────────────────

async def get_subsidence_timeseries(
    lng: float,
    lat: float,
    db:  AsyncSession,
    radius_m: float = 30.0,
) -> Optional[dict]:
    """
    Retorna la serie temporal de subsidencia para el punto GPS (lng, lat).
    Promedia los píxeles InSAR dentro de radius_m metros.

    Retorna None si no hay datos InSAR disponibles en la zona.
    """
    sql = text("""
        SELECT acquisition_date, AVG(vert_mm) AS vert_mm
        FROM insar_displacement
        WHERE ST_DWithin(
            location::geography,
            ST_SetSRID(ST_Point(:lng, :lat), 4326)::geography,
            :radius_m
        )
        GROUP BY acquisition_date
        ORDER BY acquisition_date ASC
    """)
    result = await db.execute(sql, {"lng": lng, "lat": lat, "radius_m": radius_m})
    rows   = result.fetchall()

    if not rows:
        return None

    dates  = [r.acquisition_date.isoformat() for r in rows]
    values = np.array([r.vert_mm for r in rows], dtype=np.float64)

    # Velocidad (mm/año) por regresión lineal
    n     = len(values)
    x     = np.arange(n, dtype=np.float64)
    slope = np.polyfit(x, values, 1)[0]
    # 1 intervalo = 12 días → pendiente en mm/intervalo → mm/año
    velocity_mm_yr = slope * (365.25 / 12.0)

    # Aceleración: pendiente de la velocidad instantánea
    velocities = np.diff(values)   # mm/intervalo
    accel_slope = np.polyfit(np.arange(len(velocities)), velocities, 1)[0] if len(velocities) > 2 else 0.0
    accel_mm_yr2 = accel_slope * (365.25 / 12.0) ** 2

    return {
        "dates":          dates,
        "displacement_mm": values.tolist(),
        "velocity_mm_yr":  round(velocity_mm_yr, 2),
        "acceleration":    round(accel_mm_yr2, 4),
        "n_acquisitions":  n,
        "total_subsidence_mm": round(float(values[-1] - values[0]), 1),
    }


async def get_velocity_map(
    bbox: tuple[float, float, float, float],
    db:   AsyncSession,
) -> list[dict]:
    """
    Retorna velocidades de subsidencia para todos los píxeles en el bbox.
    bbox = (min_lng, min_lat, max_lng, max_lat)
    Usa la vista materializada insar_velocity para eficiencia.
    """
    sql = text("""
        SELECT lng, lat, velocity_mm_yr, n_acquisitions
        FROM insar_velocity
        WHERE ST_Within(
            ST_SetSRID(ST_Point(lng, lat), 4326),
            ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326)
        )
    """)
    result = await db.execute(sql, {
        "min_lng": bbox[0], "min_lat": bbox[1],
        "max_lng": bbox[2], "max_lat": bbox[3],
    })
    return [dict(r._mapping) for r in result.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# SCRIPT DE INGESTA BATCH (ejecutar desde línea de comandos)
# ──────────────────────────────────────────────────────────────────────────────

async def ingest_all(insar_dir: Path, db: AsyncSession, subsample: int = 4):
    """
    Ingesta todos los GeoTIFFs en insar_dir/.
    Asume que el nombre del archivo contiene la fecha en formato YYYYMMDD.
    Ejemplo: toluca_20240115.tif → date(2024, 1, 15)
    """
    import re
    for tif in sorted(insar_dir.glob("*.tif")):
        match = re.search(r"(\d{8})", tif.stem)
        if not match:
            log.warning("No se pudo extraer fecha de: %s", tif.name)
            continue
        date_str = match.group(1)
        acq_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        await ingest_insar_tif(str(tif), acq_date, db, subsample=subsample)
```

---

## api/risk.py

```python
"""
api/risk.py
===========
Cálculo del índice de riesgo integrado R(x,y,t) ∈ [0, 10].

Combina:
  - Severidad de la grieta (de Fase 7)
  - Velocidad de subsidencia InSAR
  - Aceleración de subsidencia
  - Densidad de reportes en la zona

Fórmula:
  R = α·severity_score + β·|ṡ_norm| + γ·max(s̈_norm, 0) + δ·ρ_norm

  Donde los términos _norm están normalizados a [0, 10].
"""

from api.models import Report

# Pesos del índice de riesgo (α + β + γ + δ = 1.0)
# Ajustar con datos reales tras validación con inspectores municipales
ALPHA = 0.40   # peso de la severidad de la grieta
BETA  = 0.30   # peso de la velocidad de subsidencia
GAMMA = 0.20   # peso de la aceleración de subsidencia
DELTA = 0.10   # peso de la densidad de reportes ciudadanos

# Valores de normalización (ajustar con estadísticas del Valle de Toluca)
MAX_SUBSID_VEL_MM_YR  = 15.0   # velocidad de subsidencia "máxima" esperada
MAX_SUBSID_ACCEL      = 3.0    # aceleración "máxima" esperada (mm/año²)
MAX_REPORT_DENSITY    = 5.0    # reportes por 50m de radio = "zona caliente"


def compute_risk_index(report: Report) -> float:
    """
    Calcula R(x,y,t) para un reporte individual.

    Los términos de subsidencia pueden ser None si aún no hay datos InSAR
    disponibles para esa zona; en ese caso solo se usa la severidad de la grieta.
    """
    # Componente 1: severidad de la grieta [0–10]
    sev_score = getattr(report, "severity_score", None)
    if sev_score is None:
        # Mapear clase a puntuación si solo hay clase
        sev_map = {"leve": 2.0, "moderado": 5.0, "severo": 9.0}
        sev_score = sev_map.get(getattr(report, "severity_class", "leve"), 2.0)

    # Componente 2: velocidad de subsidencia [0–10]
    vel = getattr(report, "subsid_vel_mm_yr", None)
    if vel is not None:
        vel_norm = min(abs(vel) / MAX_SUBSID_VEL_MM_YR * 10.0, 10.0)
    else:
        vel_norm = None

    # Componente 3: aceleración de subsidencia [0–10]
    accel = getattr(report, "subsid_accel", None)
    if accel is not None:
        accel_norm = min(max(accel, 0) / MAX_SUBSID_ACCEL * 10.0, 10.0)
    else:
        accel_norm = None

    # Si no hay datos InSAR, redistribuir los pesos entre lo disponible
    if vel_norm is None or accel_norm is None:
        return round(min(sev_score, 10.0), 2)

    # Componente 4: densidad de reportes (se calculará en la API con una consulta
    # espacial; por ahora usar 0 hasta implementar esa consulta)
    rho_norm = 0.0

    R = (ALPHA * sev_score +
         BETA  * vel_norm  +
         GAMMA * accel_norm +
         DELTA * rho_norm)

    return round(min(R, 10.0), 2)


def risk_label(risk_index: float) -> dict:
    """Convierte R numérico a semáforo y prioridad de mantenimiento."""
    if risk_index < 3.0:
        return {"color": "verde",     "prioridad": "baja",
                "accion": "Monitorear en próxima revisión programada"}
    elif risk_index < 6.0:
        return {"color": "amarillo",  "prioridad": "media",
                "accion": "Bacheo preventivo en los próximos 90 días"}
    elif risk_index < 8.0:
        return {"color": "naranja",   "prioridad": "alta",
                "accion": "Intervención en los próximos 30 días"}
    else:
        return {"color": "rojo",      "prioridad": "crítica",
                "accion": "Intervención inmediata — posible peligro para vehículos"}
```

---

## Endpoint de API para subsidencia (añadir a api/main.py)

```python
@app.get("/api/v1/subsidencia")
async def get_subsidencia(
    lat: float, lng: float,
    db:  AsyncSession = Depends(get_db),
):
    """
    Serie temporal de subsidencia InSAR para un punto GPS.
    Radio de búsqueda: 30 m (2 píxeles a 15 m/px).
    """
    from api.insar import get_subsidence_timeseries
    data = await get_subsidence_timeseries(lng, lat, db)
    if data is None:
        raise HTTPException(404, "No hay datos InSAR para este punto")
    return data


@app.get("/api/v1/riesgo")
async def get_riesgo(
    lat: float, lng: float, radio_m: float = 500,
    db:  AsyncSession = Depends(get_db),
):
    """
    Retorna reportes con su índice de riesgo en el radio indicado,
    ordenados por riesgo descendente.
    """
    from sqlalchemy import text
    from api.risk import risk_label
    sql = text("""
        SELECT id, created_at, severity_class, risk_index,
               crack_width_mm, subsid_vel_mm_yr,
               ST_X(location::geometry) as lng,
               ST_Y(location::geometry) as lat
        FROM reports
        WHERE categoria = 'Aprobado'
          AND risk_index IS NOT NULL
          AND ST_DWithin(
              location::geography,
              ST_SetSRID(ST_Point(:lng, :lat), 4326)::geography,
              :radio_m
          )
        ORDER BY risk_index DESC
        LIMIT 100
    """)
    rows = await db.execute(sql, {"lat": lat, "lng": lng, "radio_m": radio_m})
    result = []
    for r in rows.fetchall():
        d = dict(r._mapping)
        d["semaforo"] = risk_label(d["risk_index"] or 0)
        result.append(d)
    return result
```

---

## Script de ingesta batch desde línea de comandos

```bash
# Ejecutar la primera vez para ingestar todos los GeoTIFFs disponibles
# (puede tardar horas si hay cientos de imágenes)

python - << 'EOF'
import asyncio
from pathlib import Path
from api.database import AsyncSessionLocal
from api.insar import ingest_all

async def main():
    async with AsyncSessionLocal() as db:
        await ingest_all(Path("insar_data"), db, subsample=4)
        # Refrescar vista materializada de velocidades
        await db.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY insar_velocity")
        await db.commit()
    print("Ingesta completada.")

asyncio.run(main())
EOF
```

---

## Verificación de la fase

```bash
# 1. Verificar que hay datos en la tabla
psql -U grietas -d grietas_db -c "
  SELECT acquisition_date, COUNT(*) as n_pixeles
  FROM insar_displacement
  GROUP BY acquisition_date
  ORDER BY acquisition_date DESC
  LIMIT 5;"

# 2. Consultar subsidencia en el centro de Toluca (coordenadas aproximadas)
curl "http://localhost:8000/api/v1/subsidencia?lat=19.293&lng=-99.653"
# Esperado: {"dates": [...], "displacement_mm": [...], "velocity_mm_yr": -8.5, ...}

# 3. Consultar reportes con riesgo en el centro
curl "http://localhost:8000/api/v1/riesgo?lat=19.293&lng=-99.653&radio_m=1000"
# Esperado: lista ordenada por risk_index con semáforo de colores
```

---

## Siguiente paso

**Fase 9** — Dashboard municipal (`fase9.md`): visualización del mapa de riesgo,
datos de subsidencia y evolución temporal de grietas para el equipo de
mantenimiento de vialidades de Toluca.
