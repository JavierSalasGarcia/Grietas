"""
Aplicación FastAPI principal — Grietas Toluca Backend.
"""
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from geoalchemy2.elements import WKTElement
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import IMAGES_DIR
from api.database import Base, engine, get_db
from api.models import Report
from api.schemas import ReportDetail, ReportResponse
from api.tasks import process_report


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Grietas Toluca API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir en producción
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /api/v1/reports  — recibe imagen + sidecar JSON de la PWA
# ---------------------------------------------------------------------------

@app.post("/api/v1/reports", response_model=ReportResponse, status_code=202)
async def create_report(
    image: UploadFile = File(...),
    metadata: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Guarda imagen en disco, crea registro en DB y encola procesamiento."""
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(400, "metadata no es JSON válido")

    report_id = uuid.uuid4()
    image_path = IMAGES_DIR / f"{report_id}.jpg"
    content = await image.read()
    image_path.write_bytes(content)

    gps = meta.get("gps")
    location = None
    if gps and gps.get("lat") is not None and gps.get("lng") is not None:
        location = WKTElement(
            f"POINT({gps['lng']} {gps['lat']})", srid=4326
        )

    report = Report(
        id=report_id,
        image_path=str(image_path),
        metadata_json=meta,
        categoria="Pendiente",
        processing_status="queued",
        location=location,
        previous_report_id=meta.get("previous_report_id"),
    )
    db.add(report)
    await db.commit()

    process_report.delay(str(report_id))

    return ReportResponse(
        report_id=report_id,
        status="queued",
        categoria=None,
        confianza=None,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/reports/{report_id}  — detalle de un reporte
# ---------------------------------------------------------------------------

@app.get("/api/v1/reports/{report_id}", response_model=ReportDetail)
async def get_report(
    report_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(404, "Reporte no encontrado")
    return ReportDetail(
        report_id=report.id,
        status=report.processing_status,
        categoria=report.categoria,
        confianza=report.confianza,
        created_at=report.created_at,
        crack_width_mm=report.crack_width_mm,
        crack_length_mm=report.crack_length_mm,
        crack_area_mm2=report.crack_area_mm2,
        severity_class=report.severity_class,
        risk_index=report.risk_index,
        subsid_vel_mm_yr=report.subsid_vel_mm_yr,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/reports  — lista reportes en radio geográfico
# ---------------------------------------------------------------------------

@app.get("/api/v1/reports")
async def list_reports(
    lat: float,
    lng: float,
    radio_m: float = 500,
    db: AsyncSession = Depends(get_db),
):
    """Lista reportes Aprobados dentro de un radio geográfico (metros)."""
    from sqlalchemy import text

    sql = text("""
        SELECT id, created_at, categoria, confianza,
               crack_width_mm, severity_class, risk_index,
               ST_X(location::geometry) AS lng,
               ST_Y(location::geometry) AS lat
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
    result = await db.execute(sql, {"lat": lat, "lng": lng, "radio_m": radio_m})
    return [dict(r._mapping) for r in result.fetchall()]


# ---------------------------------------------------------------------------
# GET /health  — liveness probe
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}
