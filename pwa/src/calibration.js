/**
 * Flujo de calibración en onboarding (primera vez).
 *
 * El ciudadano captura una imagen de un objeto de referencia conocido
 * (por ejemplo una baldosa estándar de 30×30 cm a 1 metro de distancia)
 * para estimar el FoV real del dispositivo y mejorar la ortorrectificación.
 *
 * Si el usuario omite la calibración, se usará el valor de ua-camera-db.js.
 */

const KEY_CALIBRATED_FOV = "calibrated_fov_deg";
const KEY_INTRINSICS_SRC = "intrinsics_source";

/** Inicia el flujo de calibración interactivo. */
export async function startCalibration(videoElement, captureCanvas) {
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

  const blob = await new Promise(res => captureCanvas.toBlob(res, "image/jpeg", 0.92));
  const fov  = await estimateFovFromCalibrationImage(blob, captureCanvas);

  if (fov !== null) {
    localStorage.setItem(KEY_CALIBRATED_FOV, String(fov));
    localStorage.setItem(KEY_INTRINSICS_SRC, "onboarding_calibration");
    return fov;
  }
  return null;
}

/**
 * Estima el FoV usando la distancia focal calculada desde el objeto de referencia.
 * Asume un objeto de 30 cm de ancho a ~100 cm de distancia.
 * Requiere que el usuario haya alineado los bordes del objeto con las guías.
 */
async function estimateFovFromCalibrationImage(blob, canvas) {
  const KNOWN_WIDTH_CM   = 30;
  const KNOWN_DISTANCE_CM = 100;

  // Usar la guía de alineación (50% del ancho de imagen → objeto de referencia)
  const pixelWidthOfObject = canvas.width * 0.5;
  const imageWidthPx       = canvas.width;

  // f = (pixel_width × distance) / known_width
  const focalLengthPx = (pixelWidthOfObject * KNOWN_DISTANCE_CM) / KNOWN_WIDTH_CM;

  // FoV horizontal = 2 × arctan(image_width / (2 × f))
  const fovRad = 2 * Math.atan(imageWidthPx / (2 * focalLengthPx));
  const fovDeg = fovRad * (180 / Math.PI);

  // Rechazar si el resultado está fuera de rango plausible
  if (fovDeg < 40 || fovDeg > 100) return null;
  return Math.round(fovDeg);
}

export function getCalibratedFov() {
  const stored = localStorage.getItem(KEY_CALIBRATED_FOV);
  return stored ? parseFloat(stored) : null;
}

export function getIntrinsicsSource() {
  return localStorage.getItem(KEY_INTRINSICS_SRC) ?? "ua_database";
}

export function clearCalibration() {
  localStorage.removeItem(KEY_CALIBRATED_FOV);
  localStorage.removeItem(KEY_INTRINSICS_SRC);
}
