import numpy as np
import pytest
from api.orthorectify import (
    build_K, build_R, compute_gsd, orthorectify, gsd_uncertainty,
    PITCH_MIN_DEG, PITCH_MAX_DEG, MAX_OUTPUT_PX, DEFAULT_FOV_DEG,
)


def make_metadata(pitch=45, height_cm=170, fov=70, intrinsics="ua_database"):
    return {
        "citizen": {"height_cm": height_cm},
        "camera": {
            "orientation": {"alpha": 90, "beta": pitch, "gamma": 0, "absolute": True},
            "resolution":  {"width": 4000, "height": 3000},
            "fov_estimate_deg": fov,
            "intrinsics_source": intrinsics,
        },
    }


def test_build_K_shape():
    K = build_K(70, 4000, 3000)
    assert K.shape == (3, 3)
    assert K[0, 2] == pytest.approx(2000)   # cx = W/2
    assert K[1, 2] == pytest.approx(1500)   # cy = H/2


def test_build_K_focal_length():
    K = build_K(70, 4000, 3000)
    # fx = (W/2) / tan(FoV/2) — verify it's positive and plausible
    assert K[0, 0] > 0
    assert K[0, 0] == K[1, 1]   # square pixels


def test_build_R_shape():
    R = build_R(0, 0, 0)
    assert R.shape == (3, 3)


def test_build_R_is_rotation():
    R = build_R(0, 0, 0)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_build_R_identity_at_zero():
    R = build_R(0, 0, 0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_gsd_45deg_170cm():
    """A 45° y 170 cm de altura el GSD debe estar en rango 0.5–1.2 mm/px."""
    gsd = compute_gsd(h_m=0.85 * 1.70, fov_h_deg=70, pitch_deg=45, width_px=4000)
    assert 0.5 <= gsd <= 1.2, f"GSD inesperado: {gsd:.3f} mm/px"


def test_gsd_increases_with_height():
    gsd_low  = compute_gsd(h_m=1.0, fov_h_deg=70, pitch_deg=45, width_px=4000)
    gsd_high = compute_gsd(h_m=2.0, fov_h_deg=70, pitch_deg=45, width_px=4000)
    assert gsd_high > gsd_low


def test_orthorectify_good_quality():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=45)
    ortho, gsd, quality = orthorectify(image, meta)
    assert quality == "good"
    assert ortho.shape[0] > 0 and ortho.shape[1] > 0
    assert 0.3 < gsd < 3.0


def test_orthorectify_output_bounded_by_max():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=45)
    ortho, _, _ = orthorectify(image, meta)
    assert ortho.shape[0] <= MAX_OUTPUT_PX
    assert ortho.shape[1] <= MAX_OUTPUT_PX


def test_orthorectify_poor_quality_at_15deg():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=15)   # fuera del rango 30°–60°
    ortho, gsd, quality = orthorectify(image, meta)
    assert quality == "poor"
    assert gsd > 0


def test_orthorectify_poor_returns_same_shape():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=10)
    ortho, _, quality = orthorectify(image, meta)
    assert quality == "poor"
    assert ortho.shape == image.shape


def test_orthorectify_boundary_pitch_min():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=PITCH_MIN_DEG)
    _, _, quality = orthorectify(image, meta)
    assert quality == "good"


def test_orthorectify_boundary_pitch_max():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = make_metadata(pitch=PITCH_MAX_DEG)
    _, _, quality = orthorectify(image, meta)
    assert quality == "good"


def test_orthorectify_default_fov_used_when_missing():
    image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    meta  = {
        "citizen": {"height_cm": 170},
        "camera": {
            "orientation": {"alpha": 0, "beta": 45, "gamma": 0},
            "resolution":  {"width": 4000, "height": 3000},
            # fov_estimate_deg omitted intentionally
        },
    }
    ortho, gsd, quality = orthorectify(image, meta)
    assert quality == "good"
    gsd_expected = compute_gsd(
        h_m=0.85 * 1.70, fov_h_deg=DEFAULT_FOV_DEG, pitch_deg=45, width_px=4000
    )
    assert gsd == pytest.approx(gsd_expected, rel=1e-6)


def test_orthorectify_empty_metadata_does_not_crash():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ortho, gsd, quality = orthorectify(image, {})
    assert ortho is not None
    assert gsd > 0


def test_gsd_uncertainty_ua_database():
    lo, hi = gsd_uncertainty(0.75, "ua_database")
    assert lo < 0.75 < hi
    assert abs((hi - lo) / 0.75 - 0.24) < 0.01   # ±12% → rango total 24%


def test_gsd_uncertainty_webxr():
    lo, hi = gsd_uncertainty(1.0, "webxr")
    assert abs((hi - lo) / 1.0 - 0.04) < 0.001   # ±2% → rango total 4%


def test_gsd_uncertainty_unknown_source_uses_18pct():
    lo, hi = gsd_uncertainty(1.0, "unknown_source")
    assert abs((hi - lo) / 1.0 - 0.36) < 0.001   # ±18% → rango total 36%


def test_gsd_uncertainty_ordering():
    for src in ("webxr", "onboarding_calibration", "ua_database", "prior"):
        lo, hi = gsd_uncertainty(1.0, src)
        assert lo < hi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
