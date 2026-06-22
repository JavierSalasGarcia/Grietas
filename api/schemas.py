from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ReportCreate(BaseModel):
    schema_version: str
    captured_at: str
    citizen: dict
    camera: dict
    gps: Optional[dict] = None
    previous_report_id: Optional[str] = None
    capture_conditions: dict


class ReportResponse(BaseModel):
    report_id: UUID
    status: str
    categoria: Optional[str] = None
    confianza: Optional[float] = None

    model_config = {"from_attributes": True}


class ReportDetail(ReportResponse):
    created_at: datetime
    crack_width_mm: Optional[float] = None
    crack_length_mm: Optional[float] = None
    crack_area_mm2: Optional[float] = None
    severity_class: Optional[str] = None
    risk_index: Optional[float] = None
    subsid_vel_mm_yr: Optional[float] = None
