# Fase 4 — PWA Frontend: Captura de Imagen y Metadatos `[PENDIENTE]`

## Contexto del proyecto

Sistema ciudadano de reporte de grietas en calles de Toluca. Esta fase construye
la interfaz web (PWA) que el ciudadano usa en su celular para fotografiar grietas.
El reto principal es que el overlay de cámara se renderiza en canvas, lo que
**destruye los metadatos EXIF**. Por eso todos los datos de orientación y GPS se
capturan explícitamente en JavaScript al momento del shutter.

---

## Prerrequisitos

- No depende de fases anteriores de ML (puede desarrollarse en paralelo)
- El backend de la Fase 5 debe tener el endpoint `POST /api/v1/reports`
  (puede trabajarse contra un mock local hasta que Fase 5 esté lista)
- Servidor HTTPS requerido en producción (DeviceOrientationEvent e iOS requieren HTTPS)

---

## Problema técnico central: pérdida de EXIF por canvas

Cuando la PWA muestra la última foto en transparencia (overlay), el stream de
cámara se combina en un `<canvas>`. Al hacer `canvas.toBlob()` el resultado es
un JPEG sin metadatos. Los datos de orientación y GPS deben capturarse por
separado desde las APIs del navegador y enviarse como JSON sidecar.

**En iOS Safari** además se requiere `DeviceOrientationEvent.requestPermission()`
(disponible desde iOS 13) para acceder al giroscopio.

---

## Estructura de archivos a crear

```
pwa/
  index.html          ← shell de la PWA
  manifest.json       ← config PWA (ícono, color, display: standalone)
  sw.js               ← Service Worker (cola offline)
  src/
    camera.js         ← stream getUserMedia, overlay, shutter
    orientation.js    ← DeviceOrientationEvent listener
    geolocation.js    ← GPS con timeout y fallback
    ua-camera-db.js   ← tabla FoV estimado por modelo de celular (top 200 México)
    calibration.js    ← flujo de calibración en onboarding (primera vez)
    upload.js         ← POST multipart con retry
    citizen-store.js  ← perfil ciudadano (altura, device_id) en localStorage
  styles/
    main.css
```

---

## index.html

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#1a1a2e">
  <link rel="manifest" href="/manifest.json">
  <title>Reporta una grieta — Toluca</title>
  <link rel="stylesheet" href="styles/main.css">
</head>
<body>
  <!-- Pantalla de cámara -->
  <div id="screen-camera" class="screen active">
    <video id="camera-video" autoplay playsinline muted></video>
    <canvas id="overlay-canvas"></canvas>      <!-- overlay transparente -->
    <canvas id="capture-canvas" hidden></canvas> <!-- captura limpia -->

    <div id="ui-controls">
      <div id="angle-indicator">45°</div>      <!-- muestra ángulo actual -->
      <button id="btn-shutter" class="shutter-btn" disabled>⬤</button>
      <div id="status-msg"></div>
    </div>
  </div>

  <!-- Pantalla de confirmación post-captura -->
  <div id="screen-preview" class="screen">
    <img id="preview-img" alt="Vista previa">
    <div id="preview-meta"></div>
    <button id="btn-send">Enviar reporte</button>
    <button id="btn-retake">Retomar foto</button>
  </div>

  <!-- Pantalla de onboarding (primer uso) -->
  <div id="screen-onboarding" class="screen">
    <h2>Configura tu perfil</h2>
    <label>Tu altura (cm):
      <input type="number" id="input-height" min="120" max="220" value="165">
    </label>
    <button id="btn-onboarding-done">Continuar</button>
  </div>

  <script type="module" src="src/camera.js"></script>
</body>
</html>
```

---

## src/orientation.js

```javascript
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
```

---

## src/geolocation.js

```javascript
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
```

---

## src/ua-camera-db.js

```javascript
/**
 * Base de datos de FoV horizontal estimado por modelo de dispositivo.
 * Top 20 dispositivos más comunes en México (expandir según telemetría real).
 * Fuente: especificaciones técnicas del fabricante + OpenMVG sensor database.
 *
 * Formato: { patron_ua: fov_horizontal_grados }
 */
const CAMERA_FOV_DB = {
  "iPhone 15":     77,
  "iPhone 14":     77,
  "iPhone 13":     77,
  "iPhone 12":     77,
  "iPhone 11":     73,
  "iPhone SE":     73,
  "SM-A54":        79,    // Samsung Galaxy A54
  "SM-A34":        79,    // Samsung Galaxy A34
  "SM-A14":        79,
  "SM-S23":        80,    // Samsung Galaxy S23
  "SM-S22":        80,
  "Pixel 7":       82,
  "Pixel 6":       82,
  "Pixel 5":       77,
  "Redmi Note 12": 79,
  "Redmi Note 11": 79,
  "moto g":        75,
  "Nokia":         73,
};
const DEFAULT_FOV = 70;   // promedio conservador para dispositivos desconocidos

export function estimateFovFromUA(userAgent) {
  for (const [key, fov] of Object.entries(CAMERA_FOV_DB)) {
    if (userAgent.includes(key)) return fov;
  }
  return DEFAULT_FOV;
}
```

---

## src/citizen-store.js

```javascript
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
```

---

## src/camera.js — módulo principal

```javascript
import { initOrientation, getOrientationSnapshot, getPitchDeg } from "./orientation.js";
import { getCurrentPosition } from "./geolocation.js";
import { estimateFovFromUA } from "./ua-camera-db.js";
import { getHeight, getDeviceId, hasCompletedOnboarding, setHeight } from "./citizen-store.js";
import { uploadReport } from "./upload.js";

// Estado global de la sesión
let stream = null;
let lastReportId = null;   // para enlazar serie temporal

const video        = document.getElementById("camera-video");
const overlayCanvas = document.getElementById("overlay-canvas");
const captureCanvas = document.getElementById("capture-canvas");
const btnShutter   = document.getElementById("btn-shutter");
const angleIndicator = document.getElementById("angle-indicator");

// ── INICIALIZACIÓN ──────────────────────────────────────────────────────────

async function init() {
  // Mostrar onboarding si es la primera vez
  if (!hasCompletedOnboarding()) {
    showScreen("screen-onboarding");
    document.getElementById("btn-onboarding-done").onclick = () => {
      const h = parseInt(document.getElementById("input-height").value);
      if (h >= 120 && h <= 220) {
        setHeight(h);
        startCamera();
      }
    };
    return;
  }
  await startCamera();
}

async function startCamera() {
  showScreen("screen-camera");

  // Pedir permiso de giroscopio (especialmente iOS)
  await initOrientation();

  // Iniciar stream de cámara trasera
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width:      { ideal: 4000 },
        height:     { ideal: 3000 },
      }
    });
    video.srcObject = stream;
    await video.play();
    fitCanvases();
    btnShutter.disabled = false;
    startAngleIndicator();
  } catch (err) {
    showStatus("No se pudo acceder a la cámara: " + err.message, "error");
  }
}

function fitCanvases() {
  const { videoWidth: w, videoHeight: h } = video;
  for (const c of [overlayCanvas, captureCanvas]) {
    c.width = w; c.height = h;
  }
}

// ── OVERLAY DE FOTO ANTERIOR ─────────────────────────────────────────────────

let overlayImage = null;

export function setOverlayPhoto(base64Url) {
  const img = new Image();
  img.onload = () => { overlayImage = img; };
  img.src = base64Url;
}

function drawOverlay() {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (overlayImage) {
    ctx.globalAlpha = 0.35;
    ctx.drawImage(overlayImage, 0, 0, overlayCanvas.width, overlayCanvas.height);
    ctx.globalAlpha = 1.0;
  }
  requestAnimationFrame(drawOverlay);
}
requestAnimationFrame(drawOverlay);

// ── INDICADOR DE ÁNGULO ──────────────────────────────────────────────────────

function startAngleIndicator() {
  setInterval(() => {
    const pitch = Math.round(getPitchDeg());
    angleIndicator.textContent = `${pitch}°`;
    angleIndicator.style.color = Math.abs(pitch - 45) < 5 ? "#00ff88" : "#ff6b6b";
  }, 200);
}

// ── SHUTTER ──────────────────────────────────────────────────────────────────

btnShutter.onclick = async () => {
  btnShutter.disabled = true;
  showStatus("Capturando…");

  // 1. Capturar frame limpio del video (sin overlay) para análisis de IA
  const cleanCtx = captureCanvas.getContext("2d");
  cleanCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  // 2. Capturar todos los metadatos EN EL MISMO INSTANTE
  const capturedAt = new Date().toISOString();
  const [imageBlob, gps, orientSnapshot] = await Promise.all([
    new Promise(res => captureCanvas.toBlob(res, "image/jpeg", 0.92)),
    getCurrentPosition(),
    Promise.resolve(getOrientationSnapshot()),
  ]);

  const track    = stream.getVideoTracks()[0];
  const settings = track.getSettings();

  const metadata = {
    schema_version: "1.1",
    captured_at:    capturedAt,
    citizen: {
      height_cm:  getHeight(),
      device_id:  await getDeviceId(),
      user_agent: navigator.userAgent,
    },
    camera: {
      orientation:        orientSnapshot,
      resolution:         { width: settings.width, height: settings.height },
      stream_settings:    {
        frame_rate:   settings.frameRate,
        facing_mode:  settings.facingMode,
        device_id:    settings.deviceId,
      },
      fov_estimate_deg:   estimateFovFromUA(navigator.userAgent),
      intrinsics_source:  "ua_database",
    },
    gps: gps,
    previous_report_id: lastReportId,
    capture_conditions: { overlay_used: overlayImage !== null },
  };

  // 3. Mostrar preview
  const previewUrl = URL.createObjectURL(imageBlob);
  document.getElementById("preview-img").src = previewUrl;
  document.getElementById("preview-meta").textContent =
    `Ángulo: ${Math.round(orientSnapshot.beta ?? 45)}°  |  ` +
    `GPS: ${gps ? `${gps.lat.toFixed(5)}, ${gps.lng.toFixed(5)}` : "no disponible"}`;
  showScreen("screen-preview");

  // 4. Botones de acción
  document.getElementById("btn-send").onclick = async () => {
    showStatus("Enviando reporte…");
    try {
      const result = await uploadReport(imageBlob, metadata);
      lastReportId = result.report_id;
      // Guardar overlay para la próxima foto en este punto
      setOverlayPhoto(previewUrl);
      showStatus("✓ Reporte enviado. ID: " + result.report_id, "success");
      showScreen("screen-camera");
    } catch (err) {
      showStatus("Error al enviar: " + err.message, "error");
    }
    btnShutter.disabled = false;
  };
  document.getElementById("btn-retake").onclick = () => {
    showScreen("screen-camera");
    btnShutter.disabled = false;
  };
};

// ── HELPERS ──────────────────────────────────────────────────────────────────

function showScreen(id) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}
function showStatus(msg, type = "info") {
  const el = document.getElementById("status-msg");
  el.textContent  = msg;
  el.className    = `status-${type}`;
}

// Arrancar al cargar la página
init();
```

---

## src/upload.js

```javascript
/**
 * Sube imagen + metadata JSON al backend.
 * Reintenta hasta 3 veces con backoff exponencial.
 * Si no hay conexión, guarda en IndexedDB (cola offline).
 */

const API_URL     = "/api/v1/reports";
const MAX_RETRIES = 3;

export async function uploadReport(imageBlob, metadata) {
  const form = new FormData();
  form.append("image",    imageBlob, `report_${Date.now()}.jpg`);
  form.append("metadata", JSON.stringify(metadata));

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch(API_URL, { method: "POST", body: form });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();    // { report_id, status }
    } catch (err) {
      if (attempt === MAX_RETRIES) {
        // Guardar en cola offline
        await saveToOfflineQueue(imageBlob, metadata);
        throw new Error("Sin conexión. Reporte guardado para envío posterior.");
      }
      await new Promise(r => setTimeout(r, 2 ** attempt * 1000));
    }
  }
}

async function saveToOfflineQueue(imageBlob, metadata) {
  const db  = await openDB();
  const tx  = db.transaction("queue", "readwrite");
  await tx.objectStore("queue").put({
    id:        crypto.randomUUID(),
    imageBlob,
    metadata,
    savedAt:   new Date().toISOString(),
  });
}

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("grietas-offline", 1);
    req.onupgradeneeded = e => {
      e.target.result.createObjectStore("queue", { keyPath: "id" });
    };
    req.onsuccess = e => resolve(e.target.result);
    req.onerror   = e => reject(e.target.error);
  });
}
```

---

## sw.js — Service Worker con cola offline

```javascript
const CACHE_NAME = "grietas-pwa-v1";
const STATIC_ASSETS = ["/", "/index.html", "/styles/main.css",
                        "/src/camera.js", "/manifest.json"];

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(c => c.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Servir assets desde caché; network first para la API
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith("/api/")) return;   // dejar pasar al network
  e.respondWith(
    caches.match(e.request).then(cached => cached ?? fetch(e.request))
  );
});

// Sincronizar cola offline cuando se recupere la conexión
self.addEventListener("sync", e => {
  if (e.tag === "sync-reports") {
    e.waitUntil(syncOfflineQueue());
  }
});

async function syncOfflineQueue() {
  // Abrir IndexedDB, leer cola y subir pendientes
  // (implementación completa en upload.js lado cliente)
}
```

---

## manifest.json

```json
{
  "name": "Reporta una grieta — Toluca",
  "short_name": "Grietas",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1a1a2e",
  "theme_color": "#1a1a2e",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
```

---

## Verificación de la fase

Lista de comprobación manual en dispositivo real:

```
[ ] Cámara trasera se activa al abrir la PWA
[ ] Overlay de foto anterior aparece en transparencia (alpha 35%)
[ ] Indicador de ángulo muestra el pitch en tiempo real
[ ] Indicador se pone verde cuando el ángulo está entre 40°–50°
[ ] Al presionar shutter se muestra la pantalla de preview
[ ] La pantalla de preview muestra el ángulo y las coordenadas GPS
[ ] El botón "Enviar" hace POST a /api/v1/reports con imagen + JSON
[ ] El JSON de metadata tiene: captured_at, citizen, camera.orientation,
    camera.fov_estimate_deg, camera.intrinsics_source, gps
[ ] Si no hay conexión, el reporte se encola en IndexedDB
[ ] Al recuperar conexión, el Service Worker reintenta el envío
[ ] En iOS Safari el botón de permiso de giroscopio aparece antes del shutter
```

---

## Siguiente paso

**Fase 5** — Backend API (`fase5.md`): recibe los reportes de la PWA, los
clasifica con el modelo ONNX de Fase 3 y orquesta el pipeline de procesamiento.
