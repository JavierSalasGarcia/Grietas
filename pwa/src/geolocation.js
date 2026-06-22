/** Obtiene posición GPS con alta precisión y timeout de 10 segundos. */
export function getCurrentPosition() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      resolve(null);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      pos => resolve({
        lat:        pos.coords.latitude,
        lng:        pos.coords.longitude,
        alt_m:      pos.coords.altitude,
        accuracy_m: pos.coords.accuracy,
      }),
      err => {
        console.warn("GPS no disponible:", err.message);
        resolve(null);    // no bloquear el flujo si GPS falla
      },
      { enableHighAccuracy: true, timeout: 10_000, maximumAge: 0 }
    );
  });
}
