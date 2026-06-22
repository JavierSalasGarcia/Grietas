/** Perfil del ciudadano persistido en localStorage. */
const KEY_HEIGHT    = "citizen_height_cm";
const KEY_DEVICE_ID = "citizen_device_id";

export function getHeight() {
  return parseInt(localStorage.getItem(KEY_HEIGHT) ?? "165", 10);
}
export function setHeight(cm) {
  localStorage.setItem(KEY_HEIGHT, String(cm));
}
export function hasCompletedOnboarding() {
  return localStorage.getItem(KEY_HEIGHT) !== null;
}

export async function getDeviceId() {
  let id = localStorage.getItem(KEY_DEVICE_ID);
  if (!id) {
    // Generar ID único persistente (no cambia entre sesiones)
    const arr = new Uint8Array(16);
    crypto.getRandomValues(arr);
    id = Array.from(arr).map(b => b.toString(16).padStart(2, "0")).join("");
    localStorage.setItem(KEY_DEVICE_ID, id);
  }
  return id;
}
