import uuid
from datetime import datetime, timezone

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from api.database import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    image_path = Column(Text, nullable=False)
    metadata_json = Column(JSONB, nullable=False)

    categoria = Column(String(20))
    # "Aprobado" | "No Aprobado" | "Pendiente"
    confianza = Column(Float)
    processing_status = Column(String(20), default="queued")
    # "queued" → "classifying" → "orthorectifying" → "segmenting" → "done" | "error"
    error_message = Column(Text)

    location = Column(Geometry("POINT", srid=4326))

    crack_width_mm = Column(Float)
    crack_length_mm = Column(Float)
    crack_area_mm2 = Column(Float)
    crack_fractal = Column(Float)
    crack_angle_deg = Column(Float)
    crack_area_ratio = Column(Float)
    crack_branches = Column(Integer)
    severity_class = Column(String(10))
    # "leve" | "moderado" | "severo"
    gsd_mm_px = Column(Float)

    subsid_mm = Column(Float)
    subsid_vel_mm_yr = Column(Float)
    subsid_accel = Column(Float)

    risk_index = Column(Float)

    previous_report_id = Column(
        UUID(as_uuid=True), ForeignKey("reports.id"), nullable=True
    )
    delta_width_mm = Column(Float)
    delta_length_mm = Column(Float)
    growth_rate_mm_mo = Column(Float)

    intrinsics_source = Column(String(30))
    # "webxr" | "calibration" | "ua_database" | "prior"
    ortho_quality = Column(String(10))
    # "good" | "poor"
