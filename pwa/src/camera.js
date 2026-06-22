import { initOrientation, getOrientationSnapshot, getPitchDeg } from "./orientation.js";
import { getCurrentPosition } from "./geolocation.js";
import { estimateFovFromUA } from "./ua-camera-db.js";
import { getHeight, getDeviceId, hasCompletedOnboarding, setHeight } from "./citizen-store.js";
import { getCalibratedFov, getIntrinsicsSource } from "./calibration.js";
import { uploadReport } from "./upload.js";

// Estado global de la sesión
let stream = null;
let lastReportId = null;
let angleIntervalId = null;

const video          = document.getElementById("camera-video");
const overlayCanvas  = document.getElementById("overlay-canvas");
const captureCanvas  = document.getElementById("capture-canvas");
const btnShutter     = document.getElementById("btn-shutter");
const angleIndicator = document.getElementById("angle-indicator");

// ── INICIALIZACIÓN ──────────────────────────────────────────────────────────

async function init() {
  if (!hasCompletedOnboarding()) {
    showScreen("screen-onboarding");
    document.getElementById("btn-onboarding-done").onclick = () => {
      const h = parseInt(document.getElementById("input-height").value, 10);
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
    requestAnimationFrame(drawOverlay);
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

// ── INDICADOR DE ÁNGULO ──────────────────────────────────────────────────────

function startAngleIndicator() {
  if (angleIntervalId) clearInterval(angleIntervalId);
  angleIntervalId = setInterval(() => {
    const pitch = Math.round(getPitchDeg());
    angleIndicator.textContent = `${pitch}°`;
    // Verde cuando está entre 40°–50° (zona óptima para foto oblicua a 45°)
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

  // Determinar fuente de intrínsecos de cámara
  const calibratedFov = getCalibratedFov();
  const intrinsicsSrc = getIntrinsicsSource();
  const fovEstimate   = calibratedFov ?? estimateFovFromUA(navigator.userAgent);

  const metadata = {
    schema_version: "1.1",
    captured_at:    capturedAt,
    citizen: {
      height_cm:  getHeight(),
      device_id:  await getDeviceId(),
      user_agent: navigator.userAgent,
    },
    camera: {
      orientation: orientSnapshot,
      resolution:  { width: settings.width, height: settings.height },
      stream_settings: {
        frame_rate:  settings.frameRate,
        facing_mode: settings.facingMode,
        device_id:   settings.deviceId,
      },
      fov_estimate_deg:  fovEstimate,
      intrinsics_source: intrinsicsSrc,
    },
    gps: gps,
    previous_report_id:  lastReportId,
    capture_conditions:  { overlay_used: overlayImage !== null },
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
  el.textContent = msg;
  el.className   = `status-${type}`;
}

// Arrancar al cargar la página
init();
