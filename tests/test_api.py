"""
Tests unitarios para el backend FastAPI (Fase 5).
Se usan mocks para evitar dependencias de ONNX, PostgreSQL y Redis.
"""
import io
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------

SAMPLE_METADATA = {
    "schema_version": "1.1",
    "captured_at": "2025-01-01T12:00:00Z",
    "citizen": {"height_cm": 170, "device_id": "test-device", "user_agent": "pytest"},
    "camera": {
        "orientation": {"alpha": 90.0, "beta": 45.0, "gamma": 0.0, "absolute": True},
        "resolution": {"width": 4000, "height": 3000},
        "fov_estimate_deg": 70.0,
        "intrinsics_source": "prior",
    },
    "gps": {"lat": 19.293, "lng": -99.653, "alt_m": 2680.0, "accuracy_m": 5.0},
    "previous_report_id": None,
    "capture_conditions": {"overlay_used": False},
}

FAKE_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # cabecera JPEG mínima


def _make_report(report_id: uuid.UUID, status: str = "done") -> MagicMock:
    r = MagicMock()
    r.id = report_id
    r.processing_status = status
    r.categoria = "Aprobado"
    r.confianza = 0.97
    r.created_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    r.crack_width_mm = 2.3
    r.crack_length_mm = 150.0
    r.crack_area_mm2 = 345.0
    r.severity_class = "moderado"
    r.risk_index = 4.2
    r.subsid_vel_mm_yr = -12.0
    return r


# ---------------------------------------------------------------------------
# Test: POST /api/v1/reports
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_report_queued():
    """Crear un reporte devuelve 202 con status=queued."""
    from api.database import get_db
    from api.main import app

    mock_report_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("api.main.process_report") as mock_task,
        patch("api.main.IMAGES_DIR") as mock_dir,
        patch("uuid.uuid4", return_value=mock_report_id),
    ):
        mock_path = MagicMock()
        mock_path.__truediv__ = MagicMock(return_value=mock_path)
        mock_path.write_bytes = MagicMock()
        mock_path.__str__ = MagicMock(return_value=f"uploads/{mock_report_id}.jpg")
        mock_dir.__truediv__ = MagicMock(return_value=mock_path)
        mock_task.delay = MagicMock()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/reports",
                files={"image": ("crack.jpg", FAKE_IMAGE, "image/jpeg")},
                data={"metadata": json.dumps(SAMPLE_METADATA)},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["categoria"] is None
    assert "report_id" in body


# ---------------------------------------------------------------------------
# Test: GET /api/v1/reports/{report_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_report_found():
    """Obtener un reporte existente devuelve sus datos."""
    from api.database import get_db
    from api.main import app

    report_id = uuid.uuid4()
    mock_report = _make_report(report_id)

    async def override_get_db():
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_report)
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/reports/{report_id}")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["categoria"] == "Aprobado"
    assert body["severity_class"] == "moderado"
    assert body["risk_index"] == pytest.approx(4.2)


@pytest.mark.asyncio
async def test_get_report_not_found():
    """Obtener un reporte inexistente devuelve 404."""
    from api.database import get_db
    from api.main import app

    async def override_get_db():
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=None)
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/reports/{uuid.uuid4()}")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Test: POST /api/v1/reports — metadata inválida
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_report_invalid_metadata():
    """Metadata no JSON devuelve 400."""
    from api.database import get_db
    from api.main import app

    async def override_get_db():
        yield AsyncMock()

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/reports",
            files={"image": ("crack.jpg", FAKE_IMAGE, "image/jpeg")},
            data={"metadata": "esto no es JSON"},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test: GET /health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Test: risk.py — cálculo del índice de riesgo
# ---------------------------------------------------------------------------

def test_risk_index_severo_con_subsidencia():
    """Un reporte severo con alta subsidencia debe tener R > 5."""
    from api.risk import compute_risk_index

    report = MagicMock()
    report.severity_class = "severo"
    report.subsid_vel_mm_yr = -40.0
    report.subsid_accel = 12.0

    r = compute_risk_index(report)
    assert r > 5.0
    assert 0.0 <= r <= 10.0


def test_risk_index_sin_datos():
    """Sin métricas de grieta ni InSAR el riesgo debe ser 0."""
    from api.risk import compute_risk_index

    report = MagicMock()
    report.severity_class = None
    report.subsid_vel_mm_yr = None
    report.subsid_accel = None

    r = compute_risk_index(report)
    assert r == 0.0


def test_risk_index_leve():
    """Grieta leve sin subsidencia tiene riesgo bajo (< 3)."""
    from api.risk import compute_risk_index

    report = MagicMock()
    report.severity_class = "leve"
    report.subsid_vel_mm_yr = None
    report.subsid_accel = None

    r = compute_risk_index(report)
    assert r < 3.0


def test_risk_index_bounds():
    """El índice de riesgo siempre está en [0, 10]."""
    from api.risk import compute_risk_index

    report = MagicMock()
    report.severity_class = "severo"
    report.subsid_vel_mm_yr = -200.0   # valor extremo
    report.subsid_accel = 999.0

    r = compute_risk_index(report)
    assert 0.0 <= r <= 10.0


# ---------------------------------------------------------------------------
# Test: classifier.py — carga defensiva cuando no existe el modelo
# ---------------------------------------------------------------------------

def test_classifier_returns_none_when_model_missing():
    """El singleton es None si no existe el modelo ONNX."""
    from api import classifier as clf_module

    if not clf_module.ONNX_MODEL_PATH.exists():
        assert clf_module.classifier is None


# ---------------------------------------------------------------------------
# Test: schemas.py — validación Pydantic
# ---------------------------------------------------------------------------

def test_report_create_schema_valid():
    from api.schemas import ReportCreate

    r = ReportCreate(**SAMPLE_METADATA)
    assert r.schema_version == "1.1"
    assert r.gps["lat"] == 19.293


def test_report_response_schema():
    from api.schemas import ReportResponse

    report_id = uuid.uuid4()
    r = ReportResponse(report_id=report_id, status="queued")
    assert r.status == "queued"
    assert r.categoria is None
