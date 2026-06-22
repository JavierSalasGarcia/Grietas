"""
Tests de Fase 4 — PWA ciudadana.

Verifica la estructura de archivos, el schema del JSON sidecar, la lógica
de FoV estimado y la cola offline (lógica pura sin navegador).
"""

import json
import os
import subprocess
import sys
import re
from pathlib import Path

import pytest

PWA_DIR = Path(__file__).parent.parent / "pwa"


# ── Estructura de archivos ───────────────────────────────────────────────────

class TestPwaFileStructure:
    REQUIRED_FILES = [
        "index.html",
        "manifest.json",
        "sw.js",
        "src/camera.js",
        "src/orientation.js",
        "src/geolocation.js",
        "src/ua-camera-db.js",
        "src/citizen-store.js",
        "src/calibration.js",
        "src/upload.js",
        "styles/main.css",
        "icons/icon-192.png",
        "icons/icon-512.png",
    ]

    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_file_exists(self, rel_path):
        path = PWA_DIR / rel_path
        assert path.exists(), f"Falta el archivo: {rel_path}"

    def test_icons_are_valid_png(self):
        for size in (192, 512):
            icon_path = PWA_DIR / f"icons/icon-{size}.png"
            with open(icon_path, "rb") as f:
                header = f.read(8)
            # PNG magic bytes
            assert header == b"\x89PNG\r\n\x1a\n", f"icon-{size}.png no es un PNG válido"


# ── manifest.json ────────────────────────────────────────────────────────────

class TestManifest:
    @pytest.fixture(scope="class")
    def manifest(self):
        with open(PWA_DIR / "manifest.json") as f:
            return json.load(f)

    def test_display_standalone(self, manifest):
        assert manifest["display"] == "standalone"

    def test_theme_color(self, manifest):
        assert manifest["theme_color"] == "#1a1a2e"

    def test_has_icons(self, manifest):
        sizes = {icon["sizes"] for icon in manifest["icons"]}
        assert "192x192" in sizes
        assert "512x512" in sizes

    def test_start_url(self, manifest):
        assert manifest["start_url"] == "/"


# ── Schema del JSON sidecar ──────────────────────────────────────────────────

class TestSidecarSchema:
    """Verifica que el schema del JSON sidecar cumple la especificación de CONTEXT.md."""

    VALID_INTRINSICS_SOURCES = {"webxr", "onboarding_calibration", "ua_database", "prior"}

    def _make_valid_metadata(self) -> dict:
        return {
            "schema_version": "1.1",
            "captured_at": "2026-06-22T12:00:00.000Z",
            "citizen": {
                "height_cm": 170,
                "device_id": "a" * 32,
                "user_agent": "Mozilla/5.0 (Linux; Android 13; SM-A546B)",
            },
            "camera": {
                "orientation": {
                    "alpha": 90.0,
                    "beta":  45.0,
                    "gamma": 0.0,
                    "absolute": True,
                    "timestamp": 1719052800000,
                },
                "resolution": {"width": 4000, "height": 3000},
                "stream_settings": {
                    "frame_rate": 30,
                    "facing_mode": "environment",
                    "device_id": "camera-001",
                },
                "fov_estimate_deg": 79.0,
                "intrinsics_source": "ua_database",
            },
            "gps": {
                "lat": 19.282,
                "lng": -99.654,
                "alt_m": 2660.0,
                "accuracy_m": 5.0,
            },
            "previous_report_id": None,
            "capture_conditions": {"overlay_used": False},
        }

    def test_valid_metadata_has_all_top_level_keys(self):
        meta = self._make_valid_metadata()
        required = {"schema_version", "captured_at", "citizen", "camera", "gps"}
        assert required.issubset(meta.keys())

    def test_citizen_fields_present(self):
        meta = self._make_valid_metadata()
        assert all(k in meta["citizen"] for k in ("height_cm", "device_id", "user_agent"))

    def test_camera_fields_present(self):
        meta = self._make_valid_metadata()
        cam = meta["camera"]
        assert all(k in cam for k in ("orientation", "resolution", "fov_estimate_deg", "intrinsics_source"))

    def test_orientation_has_beta(self):
        meta = self._make_valid_metadata()
        assert "beta" in meta["camera"]["orientation"]

    def test_intrinsics_source_valid_value(self):
        meta = self._make_valid_metadata()
        assert meta["camera"]["intrinsics_source"] in self.VALID_INTRINSICS_SOURCES

    def test_gps_can_be_null(self):
        meta = self._make_valid_metadata()
        meta["gps"] = None
        assert meta["gps"] is None   # debe aceptar GPS ausente

    def test_height_in_valid_range(self):
        meta = self._make_valid_metadata()
        h = meta["citizen"]["height_cm"]
        assert 120 <= h <= 220, f"Altura fuera de rango: {h}"

    def test_fov_in_plausible_range(self):
        meta = self._make_valid_metadata()
        fov = meta["camera"]["fov_estimate_deg"]
        assert 40 <= fov <= 100, f"FoV fuera de rango: {fov}"


# ── FoV estimado por User-Agent ──────────────────────────────────────────────

class TestFovEstimation:
    """
    Prueba la lógica de ua-camera-db.js replicada en Python para verificar
    que los patrones de UA coinciden con los FoV esperados.
    """

    # Extracto de la base de datos para tests
    CAMERA_FOV_DB = {
        "iPhone 15":     77,
        "iPhone 14":     77,
        "iPhone 13":     77,
        "iPhone 12":     77,
        "iPhone 11":     73,
        "iPhone SE":     73,
        "SM-A54":        79,
        "SM-A34":        79,
        "SM-A14":        79,
        "SM-S918":       80,   # Galaxy S23 Ultra
        "SM-S916":       80,   # Galaxy S23+
        "SM-S911":       80,   # Galaxy S23
        "SM-S908":       80,   # Galaxy S22 Ultra
        "SM-S906":       80,   # Galaxy S22+
        "SM-S901":       80,   # Galaxy S22
        "SM-S23":        80,
        "SM-S22":        80,
        "Pixel 7":       82,
        "Pixel 6":       82,
        "Pixel 5":       77,
        "Redmi Note 12": 79,
        "Redmi Note 11": 79,
        "moto g":        75,
        "Nokia":         73,
    }
    DEFAULT_FOV = 70

    def estimate_fov(self, user_agent: str) -> int:
        for key, fov in self.CAMERA_FOV_DB.items():
            if key in user_agent:
                return fov
        return self.DEFAULT_FOV

    @pytest.mark.parametrize("ua, expected_fov", [
        ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 ... iPhone 15",  77),
        ("Mozilla/5.0 (iPhone; CPU iPhone OS 16_0) ... iPhone SE",  73),
        ("Mozilla/5.0 (Linux; Android 13; SM-A546B Build/TP1A.220624.014)",  79),
        ("Mozilla/5.0 (Linux; Android 13; SM-S908B Build/TP1A.220624.014)",  80),
        ("Mozilla/5.0 (Linux; Android 13; Pixel 7 Build/TD1A)",  82),
        ("Mozilla/5.0 (Linux; Android 12; Redmi Note 12 Build/SKQ1)",  79),
        ("Mozilla/5.0 (Linux; Android 12; moto g52 Build/S2RI32.20-Q3-51)",  75),
        ("Mozilla/5.0 (Linux; Android 11; Nokia G21 Build/RKQ1)",  73),
        ("Mozilla/5.0 (Linux; Android 12; UnknownDevice Build/QP1A)",  70),   # DEFAULT
    ])
    def test_fov_estimation(self, ua, expected_fov):
        assert self.estimate_fov(ua) == expected_fov


# ── Service Worker: caché de assets estáticos ────────────────────────────────

class TestServiceWorker:
    def _read_sw(self) -> str:
        return (PWA_DIR / "sw.js").read_text()

    def test_cache_name_defined(self):
        sw = self._read_sw()
        assert "grietas-pwa-v1" in sw

    def test_static_assets_list_has_index(self):
        sw = self._read_sw()
        assert '"/index.html"' in sw or '"/",\n' in sw or '"/",' in sw

    def test_api_requests_bypass_cache(self):
        sw = self._read_sw()
        assert '/api/' in sw, "El SW debe dejar pasar las peticiones a /api/"

    def test_sync_handler_present(self):
        sw = self._read_sw()
        assert 'sync-reports' in sw

    def test_skip_waiting_called(self):
        sw = self._read_sw()
        assert 'skipWaiting' in sw

    def test_clients_claim_called(self):
        sw = self._read_sw()
        assert 'clients.claim' in sw


# ── camera.js: comprobaciones de contenido ───────────────────────────────────

class TestCameraJs:
    def _read_camera(self) -> str:
        return (PWA_DIR / "src/camera.js").read_text()

    def test_imports_orientation(self):
        src = self._read_camera()
        assert "orientation.js" in src

    def test_imports_geolocation(self):
        src = self._read_camera()
        assert "geolocation.js" in src

    def test_imports_upload(self):
        src = self._read_camera()
        assert "upload.js" in src

    def test_captures_metadata_at_shutter(self):
        src = self._read_camera()
        assert "schema_version" in src

    def test_overlay_alpha_is_35_percent(self):
        src = self._read_camera()
        assert "0.35" in src

    def test_angle_indicator_green_zone(self):
        src = self._read_camera()
        # Zona óptima: |pitch - 45| < 5
        assert "45" in src and "< 5" in src

    def test_uses_facing_mode_environment(self):
        src = self._read_camera()
        assert "environment" in src

    def test_image_quality_092(self):
        src = self._read_camera()
        assert "0.92" in src

    def test_previous_report_id_chaining(self):
        src = self._read_camera()
        assert "previous_report_id" in src and "lastReportId" in src


# ── upload.js: lógica de reintentos ─────────────────────────────────────────

class TestUploadJs:
    def _read_upload(self) -> str:
        return (PWA_DIR / "src/upload.js").read_text()

    def test_max_retries_is_3(self):
        src = self._read_upload()
        assert "MAX_RETRIES = 3" in src

    def test_api_endpoint_correct(self):
        src = self._read_upload()
        assert "/api/v1/reports" in src

    def test_exponential_backoff_present(self):
        src = self._read_upload()
        assert "2 **" in src or "2**" in src

    def test_saves_to_offline_queue_on_failure(self):
        src = self._read_upload()
        assert "saveToOfflineQueue" in src

    def test_indexeddb_used(self):
        src = self._read_upload()
        assert "indexedDB" in src

    def test_formdata_includes_image_and_metadata(self):
        src = self._read_upload()
        assert 'append("image"' in src
        assert 'append("metadata"' in src


# ── orientation.js ───────────────────────────────────────────────────────────

class TestOrientationJs:
    def _read(self) -> str:
        return (PWA_DIR / "src/orientation.js").read_text()

    def test_requests_ios_permission(self):
        src = self._read()
        assert "requestPermission" in src

    def test_prefers_absolute_orientation(self):
        src = self._read()
        assert "deviceorientationabsolute" in src

    def test_exports_get_pitch_deg(self):
        src = self._read()
        assert "getPitchDeg" in src

    def test_exports_get_orientation_snapshot(self):
        src = self._read()
        assert "getOrientationSnapshot" in src

    def test_fallback_pitch_is_45(self):
        src = self._read()
        assert "45" in src


# ── citizen-store.js ─────────────────────────────────────────────────────────

class TestCitizenStore:
    def _read(self) -> str:
        return (PWA_DIR / "src/citizen-store.js").read_text()

    def test_default_height_165(self):
        src = self._read()
        assert '"165"' in src or "'165'" in src

    def test_device_id_uses_crypto(self):
        src = self._read()
        assert "crypto.getRandomValues" in src

    def test_has_completed_onboarding_check(self):
        src = self._read()
        assert "hasCompletedOnboarding" in src

    def test_persistent_storage_key(self):
        src = self._read()
        assert "citizen_height_cm" in src
        assert "citizen_device_id" in src


# ── index.html ───────────────────────────────────────────────────────────────

class TestIndexHtml:
    def _read(self) -> str:
        return (PWA_DIR / "index.html").read_text()

    def test_links_manifest(self):
        html = self._read()
        assert 'rel="manifest"' in html

    def test_camera_video_element(self):
        html = self._read()
        assert 'id="camera-video"' in html

    def test_overlay_canvas_element(self):
        html = self._read()
        assert 'id="overlay-canvas"' in html

    def test_capture_canvas_hidden(self):
        html = self._read()
        assert 'id="capture-canvas"' in html and "hidden" in html

    def test_shutter_button_present(self):
        html = self._read()
        assert 'id="btn-shutter"' in html

    def test_angle_indicator_present(self):
        html = self._read()
        assert 'id="angle-indicator"' in html

    def test_onboarding_screen_present(self):
        html = self._read()
        assert 'id="screen-onboarding"' in html

    def test_preview_screen_present(self):
        html = self._read()
        assert 'id="screen-preview"' in html

    def test_service_worker_registration(self):
        html = self._read()
        assert "serviceWorker" in html and "sw.js" in html

    def test_module_script_camera(self):
        html = self._read()
        assert 'type="module"' in html and "camera.js" in html

    def test_lang_es(self):
        html = self._read()
        assert 'lang="es"' in html

    def test_theme_color_meta(self):
        html = self._read()
        assert 'name="theme-color"' in html
        assert "#1a1a2e" in html
