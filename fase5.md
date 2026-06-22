# Fase 5 — Backend API (FastAPI) `[PENDIENTE]`

## Contexto del proyecto

Sistema de reporte de grietas en calles de Toluca. Esta fase construye el
backend que recibe reportes de la PWA (Fase 4), clasifica las imágenes con el
modelo ONNX (Fase 3), y orquesta el pipeline completo de procesamiento:
ortorectificación (Fase 6), segmentación (Fase 7) e integración de subsidencia
(Fase 8).

---

## Prerrequisitos

- Fase 3 completada: `models/student_mobilenetv3.onnx` existe
- `models/student_config.json` existe (contiene `class_to_idx` y stats)
- `dataset_processed/dataset_stats.json` existe (mean_rgb, std_rgb)
- PostgreSQL ≥ 14 con extensión PostGIS instalada
- Redis (para la cola de tareas Celery)
- `pip install fastapi uvicorn[standard] onnxruntime sqlalchemy asyncpg geoalchemy2 psycopg2-binary celery redis python-multipart pillow numpy`

---

## Estructura de archivos a crear

```
api/
  main.py              ← app FastAPI, endpoints
  models.py            ← modelos SQLAlchemy (ORM)
  database.py          ← conexión a PostgreSQL
  classifier.py        ← inferencia con ONNX
  tasks.py             ← tareas Celery (procesamiento async)
  schemas.py           ← modelos Pydantic (request/response)
  config.py            ← variables de entorno
  alembic/             ← migraciones de base de datos
    env.py
    versions/
      001_initial.py
```

---

## api/config.py

```python
"""
Variables de configuración leídas de variables de entorno.
Usar un archivo .env en desarrollo con python-dotenv.
"""
import os
from pathlib import Path

# Base de datos
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://grietas:grietas@localhost:5432/grietas_db"
)

# Redis (para Celery)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Modelos
MODELS_DIR       = Path(os.getenv("MODELS_DIR", "models"))
ONNX_MODEL_PATH  = MODELS_DIR / "student_mobilenetv3.onnx"
STUDENT_CFG_PATH = MODELS_DIR / "student_config.json"
DATASET_STATS    = Path("dataset_processed/dataset_stats.json")

# Almacenamiento de imágenes
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "uploads"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros del clasificador
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.5"))
```

---

## api/database.py

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from api.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

---

## api/models.py

```python
"""
Modelos SQLAlchemy — mapean a tablas de PostgreSQL con PostGIS.
"""
import uuid
from datetime import datetime, timezone

from geoalchemy2 import Geometry
from sqlalchemy import (Column, DateTime, Float, ForeignKey,
                        String, Text, Integer, Boolean)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from api.database import Base


class Report(Base):
    __tablename__ = "reports"

    id              = Column(UUID(as_uuid=True), primary_key=True,
                             default=uuid.uuid4)
    created_at      = Column(DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc))

    # Archivos
    image_path      = Column(Text, nullable=False)   # ruta local al JPEG
    metadata_json   = Column(JSONB, nullable=False)  # sidecar JSON de la PWA

    # Resultado del clasificador
    categoria       = Column(String(20))   # "Aprobado" | "No Aprobado" | "Pendiente"
    confianza       = Column(Float)
    processing_status = Column(String(20), default="queued")
    #   "queued" → "classifying" → "orthorectifying" → "segmenting" → "done" | "error"
    error_message   = Column(Text)

    # Geolocalización (PostGIS Point WGS84)
    location        = Column(Geometry("POINT", srid=4326))

    # Métricas de grieta (NULL si No Aprobado)
    crack_width_mm   = Column(Float)
    crack_length_mm  = Column(Float)
    crack_area_mm2   = Column(Float)
    crack_fractal    = Column(Float)
    crack_angle_deg  = Column(Float)
    crack_area_ratio = Column(Float)
    crack_branches   = Column(Integer)
    severity_class   = Column(String(10))   # "leve" | "moderado" | "severo"
    gsd_mm_px        = Column(Float)

    # Subsidencia InSAR en el punto (se rellena en Fase 8)
    subsid_mm         = Column(Float)
    subsid_vel_mm_yr  = Column(Float)
    subsid_accel      = Column(Float)

    # Índice de riesgo integrado (se rellena al tener subsidencia)
    risk_index        = Column(Float)

    # Serie temporal: enlace con el reporte anterior del mismo punto
    previous_report_id = Column(UUID(as_uuid=True),
                                ForeignKey("reports.id"), nullable=True)
    # Cambio respecto al anterior
    delta_width_mm   = Column(Float)
    delta_length_mm  = Column(Float)
    growth_rate_mm_mo = Column(Float)   # mm/mes

    # Metadatos de calibración usados
    intrinsics_source = Column(String(30))   # "webxr"|"calibration"|"ua_database"|"prior"
    ortho_quality     = Column(String(10))   # "good" | "poor" (pitch fuera de 30-60°)
```

---

## api/schemas.py

```python
from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class ReportCreate(BaseModel):
    """Esquema del JSON de metadata enviado por la PWA."""
    schema_version: str
    captured_at:    str
    citizen:        dict
    camera:         dict
    gps:            Optional[dict]
    previous_report_id: Optional[str]
    capture_conditions: dict


class ReportResponse(BaseModel):
    report_id:  UUID
    status:     str             # "queued" | "done" | "error"
    categoria:  Optional[str]
    confianza:  Optional[float]

    class Config:
        from_attributes = True


class ReportDetail(ReportResponse):
    created_at:      datetime
    crack_width_mm:  Optional[float]
    crack_length_mm: Optional[float]
    crack_area_mm2:  Optional[float]
    severity_class:  Optional[str]
    risk_index:      Optional[float]
    subsid_vel_mm_yr: Optional[float]
```

---

## api/classifier.py

```python
"""
Clasificador binario usando el modelo ONNX del Student (Fase 3).
Se carga una sola vez al iniciar el servidor (singleton).
"""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path

from api.config import ONNX_MODEL_PATH, STUDENT_CFG_PATH, DATASET_STATS


class CrackClassifier:
    def __init__(self):
        # Cargar sesión ONNX (thread-safe para múltiples peticiones)
        self._session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        cfg = json.loads(STUDENT_CFG_PATH.read_text())
        stats = json.loads(Path(DATASET_STATS).read_text())

        # Orden alfabético de ImageFolder: 0=Aprobado, 1=No_Aprobado
        self._idx_to_class = {v: k for k, v in cfg["class_to_idx"].items()}
        self._mean = np.array(stats["mean_rgb"], dtype=np.float32)
        self._std  = np.array(stats["std_rgb"],  dtype=np.float32)

    def predict(self, image_path: str) -> dict:
        """
        Clasifica una imagen.
        Retorna {"categoria": str, "confianza": float, "probabilidades": dict}
        """
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((224, 224))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = arr.transpose(2, 0, 1)[np.newaxis]      # (1, 3, 224, 224)

        logits = self._session.run(None, {"image": arr})[0][0]
        probs  = np.exp(logits) / np.exp(logits).sum()
        idx    = int(np.argmax(probs))

        return {
            "categoria":  self._idx_to_class[idx],
            "confianza":  float(probs[idx]),
            "probabilidades": {
                self._idx_to_class[i]: float(p) for i, p in enumerate(probs)
            },
        }


# Singleton global — se inicializa al importar el módulo
classifier = CrackClassifier()
```

---

## api/tasks.py

```python
"""
Tareas Celery para procesamiento asíncrono de reportes.
Cada reporte clasificado como Aprobado pasa por ortorectificación → segmentación.
"""
from celery import Celery
from api.config import REDIS_URL

celery_app = Celery("grietas", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_serializer    = "json"
celery_app.conf.result_serializer  = "json"


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_report(self, report_id: str):
    """
    Pipeline completo para un reporte Aprobado:
    1. Clasificar (si no se hizo en tiempo real)
    2. Ortorectificar
    3. Segmentar grieta y extraer métricas
    4. Consultar subsidencia InSAR
    5. Calcular índice de riesgo
    6. Actualizar registro en DB
    """
    from api.database import AsyncSessionLocal
    from api.models import Report
    from api.classifier import classifier
    import asyncio

    async def _run():
        async with AsyncSessionLocal() as db:
            report = await db.get(Report, report_id)
            if not report:
                return

            try:
                # Paso 1 — Clasificación
                report.processing_status = "classifying"
                await db.commit()

                result = classifier.predict(report.image_path)
                report.categoria  = result["categoria"]
                report.confianza  = result["confianza"]

                if result["categoria"] == "No Aprobado":
                    report.processing_status = "done"
                    await db.commit()
                    return

                # Paso 2 — Ortorectificación
                report.processing_status = "orthorectifying"
                await db.commit()

                from api.orthorectify import orthorectify
                import cv2, numpy as np
                image = cv2.imread(report.image_path)
                meta  = report.metadata_json
                ortho, gsd = orthorectify(image, meta)
                ortho_path = report.image_path.replace(".jpg", "_ortho.jpg")
                cv2.imwrite(ortho_path, ortho)

                # Paso 3 — Segmentación y métricas
                report.processing_status = "segmenting"
                await db.commit()

                from api.crack_analysis import analyze_crack
                metrics = analyze_crack(ortho, gsd)
                report.crack_width_mm   = metrics["crack_width_mm"]
                report.crack_length_mm  = metrics["crack_length_mm"]
                report.crack_area_mm2   = metrics["crack_area_mm2"]
                report.crack_fractal    = metrics["crack_fractal_dim"]
                report.crack_angle_deg  = metrics["crack_angle_deg"]
                report.crack_area_ratio = metrics["crack_area_ratio"]
                report.crack_branches   = metrics["crack_branches"]
                report.severity_class   = metrics["severity_class"]
                report.gsd_mm_px        = metrics["gsd_mm_px"]
                report.intrinsics_source = meta.get("camera", {}).get("intrinsics_source")

                # Paso 4 — Subsidencia InSAR (si hay datos disponibles)
                if report.location:
                    from api.insar import get_subsidence_at_point
                    from geoalchemy2.shape import to_shape
                    point = to_shape(report.location)
                    subsid = await get_subsidence_at_point(point.x, point.y, db)
                    if subsid:
                        report.subsid_mm        = subsid["displacement_mm"][-1]
                        report.subsid_vel_mm_yr = subsid["velocity_mm_yr"]
                        report.subsid_accel     = subsid["acceleration"]

                # Paso 5 — Índice de riesgo
                from api.risk import compute_risk_index
                report.risk_index = compute_risk_index(report)

                report.processing_status = "done"
                await db.commit()

            except Exception as exc:
                report.processing_status = "error"
                report.error_message     = str(exc)
                await db.commit()
                raise self.retry(exc=exc)

    asyncio.run(_run())
```

---

## api/main.py

```python
"""
Aplicación FastAPI principal.
"""
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2.elements import WKTElement

from api.database import get_db, engine, Base
from api.models import Report
from api.schemas import ReportResponse, ReportDetail
from api.tasks import process_report
from api.config import IMAGES_DIR

app = FastAPI(title="Grietas Toluca API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringir en producción
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/api/v1/reports", response_model=ReportResponse, status_code=202)
async def create_report(
    image:    UploadFile = File(...),
    metadata: str        = Form(...),
    db:       AsyncSession = Depends(get_db),
):
    """
    Recibe imagen + JSON sidecar de la PWA.
    Guarda en disco, crea registro en DB y encola procesamiento asíncrono.
    """
    # Parsear y validar metadata
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(400, "metadata no es JSON válido")

    # Guardar imagen en disco
    report_id  = uuid.uuid4()
    image_path = IMAGES_DIR / f"{report_id}.jpg"
    content    = await image.read()
    image_path.write_bytes(content)

    # Extraer GPS del metadata para PostGIS
    gps = meta.get("gps")
    location = None
    if gps and gps.get("lat") and gps.get("lng"):
        location = WKTElement(f"POINT({gps['lng']} {gps['lat']})", srid=4326)

    # Crear registro en DB
    report = Report(
        id              = report_id,
        image_path      = str(image_path),
        metadata_json   = meta,
        categoria       = "Pendiente",
        processing_status = "queued",
        location        = location,
        previous_report_id = meta.get("previous_report_id"),
    )
    db.add(report)
    await db.commit()

    # Encolar tarea de procesamiento (Celery)
    process_report.delay(str(report_id))

    return ReportResponse(
        report_id = report_id,
        status    = "queued",
        categoria = None,
        confianza = None,
    )


@app.get("/api/v1/reports/{report_id}", response_model=ReportDetail)
async def get_report(report_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(404, "Reporte no encontrado")
    return report


@app.get("/api/v1/reports")
async def list_reports(
    lat: float, lng: float, radio_m: float = 500,
    desde: str = None, hasta: str = None,
    db: AsyncSession = Depends(get_db),
):
    """Lista reportes aprobados en un radio geográfico."""
    from sqlalchemy import text
    sql = text("""
        SELECT id, created_at, categoria, confianza,
               crack_width_mm, severity_class, risk_index,
               ST_X(location::geometry) as lng,
               ST_Y(location::geometry) as lat
        FROM reports
        WHERE ST_DWithin(
            location::geography,
            ST_SetSRID(ST_Point(:lng, :lat), 4326)::geography,
            :radio_m
        )
        AND categoria = 'Aprobado'
        ORDER BY created_at DESC
        LIMIT 200
    """)
    rows = await db.execute(sql, {"lat": lat, "lng": lng, "radio_m": radio_m})
    return [dict(r._mapping) for r in rows.fetchall()]


@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## Cómo levantar el backend en desarrollo

```bash
# 1. Instalar dependencias
pip install fastapi uvicorn[standard] onnxruntime sqlalchemy asyncpg \
            geoalchemy2 psycopg2-binary celery redis python-multipart \
            pillow numpy

# 2. Levantar PostgreSQL con PostGIS (Docker)
docker run -d --name grietas-pg \
  -e POSTGRES_USER=grietas \
  -e POSTGRES_PASSWORD=grietas \
  -e POSTGRES_DB=grietas_db \
  -p 5432:5432 \
  postgis/postgis:16-3.4

# 3. Levantar Redis (Docker)
docker run -d --name grietas-redis -p 6379:6379 redis:7

# 4. Iniciar la API
uvicorn api.main:app --reload --port 8000

# 5. Iniciar worker de Celery (en otra terminal)
celery -A api.tasks.celery_app worker --loglevel=info --concurrency=2
```

---

## Verificación de la fase

```bash
# Enviar un reporte de prueba
curl -X POST http://localhost:8000/api/v1/reports \
  -F "image=@test_crack.jpg" \
  -F 'metadata={"schema_version":"1.1","captured_at":"2025-01-01T12:00:00Z",
     "citizen":{"height_cm":170,"device_id":"test","user_agent":"test"},
     "camera":{"orientation":{"alpha":90,"beta":45,"gamma":0,"absolute":true},
       "resolution":{"width":4000,"height":3000},
       "fov_estimate_deg":70,"intrinsics_source":"prior"},
     "gps":{"lat":19.293,"lng":-99.653,"alt_m":2680,"accuracy_m":5},
     "previous_report_id":null,"capture_conditions":{"overlay_used":false}}'

# Consultar estado del reporte
curl http://localhost:8000/api/v1/reports/<report_id>

# Esperado tras procesamiento:
# {"report_id": "...", "status": "done",
#  "categoria": "Aprobado", "confianza": 0.97,
#  "crack_width_mm": 2.3, "severity_class": "moderado", ...}
```

---

## Siguiente paso

**Fase 6** — Módulo de ortorectificación (`fase6.md`): implementar `api/orthorectify.py`
que se invoca desde `tasks.py` para transformar la foto oblicua en vista cenital métrica.
