/**
 * Listener del giroscopio del dispositivo.
 * Captura alpha (yaw/azimut), beta (pitch), gamma (roll) continuamente.
 * Prefiere 'deviceorientationabsolute' (relativo al Norte) si está disponible.
 */

let _current = { alpha: null, beta: null, gamma: null, absolute: false };
let _initialized = false;

function _handleAbsolute(e) {
  _current = { alpha: e.alpha, beta: e.beta, gamma: e.gamma, absolute: true };
}
function _handleRelative(e) {
  if (!_current.absolute)
    _current = { alpha: e.alpha, beta: e.beta, gamma: e.gamma, absolute: false };
}

export async function initOrientation() {
  // iOS 13+ requiere permiso explícito
  if (typeof DeviceOrientationEvent?.requestPermission === "function") {
    const perm = await DeviceOrientationEvent.requestPermission();
    if (perm !== "granted") {
      console.warn("Permiso de giroscopio denegado — orientación no disponible");
      return false;
    }
  }
  window.addEventListener("deviceorientationabsolute", _handleAbsolute, true);
  window.addEventListener("deviceorientation",         _handleRelative);
  _initialized = true;
  return true;
}

/** Retorna snapshot de la orientación actual en el instante de la llamada. */
export function getOrientationSnapshot() {
  return { ..._current, timestamp: Date.now() };
}

/**
 * Devuelve el ángulo de pitch (inclinación hacia adelante/atrás).
 * beta = 0° → teléfono plano. beta = 90° → teléfono vertical (pantalla al frente).
 * Para foto a 45° desde vertical → beta ≈ 45°.
 */
export function getPitchDeg() {
  return _current.beta ?? 45;
}
