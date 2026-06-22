# Fase 9 — Dashboard Municipal `[PENDIENTE]`

## Contexto del proyecto

Sistema de reporte de grietas en calles de Toluca. Esta fase implementa el
**panel de control para el municipio**: visualización en mapa de calor del índice
de riesgo R(x,y), capa de subsidencia InSAR, ficha individual de cada reporte
(comparativo oblicua/ortofoto + métricas en mm), lista priorizada de intervenciones
y alertas automáticas por zonas con aceleración de subsidencia crítica.

---

## Prerrequisitos

- Fase 5 completada: API FastAPI con endpoints `/api/v1/reports` y `/api/v1/subsidencia`
- Fase 7 completada: vector `v_grieta` y `severity_score` en la BD
- Fase 8 completada: `risk_index` calculado y `insar_velocity` materializado
- `npm install leaflet chart.js` (o CDN)

---

## Arquitectura del dashboard

```
dashboard/
├── index.html              ← entrada única (SPA sin framework)
├── manifest.json           ← iconos, tema, display:standalone
├── sw.js                   ← cache offline para assets
├── src/
│   ├── map.js              ← Leaflet + capas WMS/GeoJSON
│   ├── heatmap.js          ← capa de riesgo R(x,y)
│   ├── insar-layer.js      ← capa de subsidencia InSAR
│   ├── report-panel.js     ← ficha individual de reporte
│   ├── priority-list.js    ← tabla priorizada
│   ├── alerts.js           ← alertas automáticas
│   ├── charts.js           ← Chart.js para series temporales
│   └── api-client.js       ← fetch wrapper con auth JWT
└── styles/
    └── dashboard.css
```

---

## Vistas principales

### Vista 1 — Mapa de Riesgo

Mapa Leaflet centrado en Toluca (19.282°N, 99.654°W, zoom 13).
Cada reporte aparece como un marcador circular cuyo color codifica `risk_index`:

| risk_index | Color hex | Prioridad |
|---|---|---|
| 8–10 | `#d32f2f` rojo | Inmediata |
| 5–8  | `#f57c00` naranja | Alta |
| 3–5  | `#fbc02d` amarillo | Media |
| 0–3  | `#388e3c` verde | Baja |

Fondo de mapa: OpenStreetMap (gratuito, sin API key). Capa base alternativa:
INEGI cartografía (WMS público).

### Vista 2 — Capa InSAR de Subsidencia

Raster de velocidad (mm/año) superpuesto con opacidad 60% sobre el mapa.
Escala de color: azul (hundimiento rápido >20mm/año) → blanco (0) → rojo
(levantamiento >5mm/año). Se actualiza en batch cuando se ingiere nueva imagen
Sentinel-1.

### Vista 3 — Ficha de Reporte

Panel lateral con:
- Miniaturas: foto original (oblicua) y ortofoto corregida, side-by-side
- Métricas `v_grieta`: longitud, ancho medio, área, dimensión fractal,
  orientación, nodos de ramificación, clase de severidad, severity_score
- Serie temporal: gráfica Chart.js de `severity_score` vs fecha para ese punto
- Serie temporal InSAR: subsidencia acumulada para el grid de 30m más cercano
- Índice de riesgo R y sus 4 componentes en barras horizontales
- Botones: Aprobar/Rechazar intervención, Exportar PDF

### Vista 4 — Lista de Prioridades

Tabla ordenada por `risk_index DESC`. Columnas:
`#`, `Dirección`, `risk_index`, `severity_class`, `subsidencia mm/año`, `Última foto`, `Acción`.
Paginada (50 registros por página). Exportable a CSV.

### Vista 5 — Alertas

Notificación en panel cuando:
- Nuevo reporte con `risk_index ≥ 8`
- Punto con `acceleration_mm_yr2 > 15` (subsidencia acelerando)
- Zona con ≥ 3 reportes severos en radio de 100 m (clúster)

Las alertas se persisten en `localStorage` hasta que el operador las marque
como atendidas.

---

## Archivos a crear

### `dashboard/index.html`

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grietas Toluca — Panel Municipal</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="styles/dashboard.css">
</head>
<body>
  <!-- Barra superior -->
  <header id="topbar">
    <span class="logo">🏙️ Grietas Toluca</span>
    <nav>
      <button class="tab active" data-view="map">Mapa</button>
      <button class="tab" data-view="priorities">Prioridades</button>
      <button class="tab" data-view="alerts">Alertas <span id="alert-badge"></span></button>
    </nav>
    <span id="last-update">Actualizando…</span>
  </header>

  <!-- Contenedor principal -->
  <main id="app">
    <!-- Vista: Mapa -->
    <div id="view-map" class="view active">
      <div id="map"></div>

      <!-- Panel lateral deslizable -->
      <aside id="report-panel" class="hidden">
        <button id="close-panel">✕</button>
        <div id="panel-content"></div>
      </aside>

      <!-- Controles de capa -->
      <div id="layer-controls">
        <label><input type="checkbox" id="toggle-insar" checked> Subsidencia InSAR</label>
        <label><input type="checkbox" id="toggle-heatmap" checked> Calor de riesgo</label>
        <input type="range" id="insar-opacity" min="0" max="100" value="60">
      </div>
    </div>

    <!-- Vista: Prioridades -->
    <div id="view-priorities" class="view hidden">
      <div id="priority-toolbar">
        <button id="export-csv">Exportar CSV</button>
        <input type="text" id="search-priority" placeholder="Buscar dirección…">
      </div>
      <table id="priority-table">
        <thead>
          <tr>
            <th>#</th><th>Dirección</th><th>Riesgo</th>
            <th>Severidad</th><th>Subsidencia mm/año</th>
            <th>Última foto</th><th>Acción</th>
          </tr>
        </thead>
        <tbody id="priority-tbody"></tbody>
      </table>
      <div id="pagination"></div>
    </div>

    <!-- Vista: Alertas -->
    <div id="view-alerts" class="view hidden">
      <ul id="alerts-list"></ul>
    </div>
  </main>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script type="module" src="src/api-client.js"></script>
  <script type="module" src="src/map.js"></script>
  <script type="module" src="src/heatmap.js"></script>
  <script type="module" src="src/insar-layer.js"></script>
  <script type="module" src="src/report-panel.js"></script>
  <script type="module" src="src/priority-list.js"></script>
  <script type="module" src="src/alerts.js"></script>
  <script type="module" src="src/charts.js"></script>
</body>
</html>
```

---

### `dashboard/src/api-client.js`

```javascript
// api-client.js — fetch wrapper con JWT desde localStorage

const API_BASE = window.API_BASE ?? "/api/v1";

async function apiFetch(path, options = {}) {
  const token = localStorage.getItem("jwt_token");
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...options.headers,
  };
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export async function getReports({ bbox, minRisk = 0, page = 1, pageSize = 50 } = {}) {
  const params = new URLSearchParams({ min_risk: minRisk, page, page_size: pageSize });
  if (bbox) params.set("bbox", bbox.join(","));
  return apiFetch(`/reports?${params}`);
}

export async function getReport(id) {
  return apiFetch(`/reports/${id}`);
}

export async function getSubsidencia({ lat, lng, radius = 30 } = {}) {
  const params = new URLSearchParams({ lat, lng, radius });
  return apiFetch(`/subsidencia?${params}`);
}

export async function getRiesgo(reportId) {
  return apiFetch(`/riesgo/${reportId}`);
}

export async function getAlerts() {
  return apiFetch("/alerts");
}
```

---

### `dashboard/src/map.js`

```javascript
// map.js — inicialización del mapa Leaflet y gestión de markers

import { getReports } from "./api-client.js";
import { openReportPanel } from "./report-panel.js";
import { renderInsarLayer } from "./insar-layer.js";

const TOLUCA = [19.282, -99.654];
const ZOOM_INIT = 13;

const RISK_COLOR = (r) => {
  if (r >= 8) return "#d32f2f";
  if (r >= 5) return "#f57c00";
  if (r >= 3) return "#fbc02d";
  return "#388e3c";
};

let map, markersLayer;

export function initMap() {
  map = L.map("map").setView(TOLUCA, ZOOM_INIT);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);

  markersLayer = L.layerGroup().addTo(map);

  renderInsarLayer(map);
  loadReports();

  // Recargar al mover el mapa (solo si bbox cambió significativamente)
  map.on("moveend", loadReports);

  // Controles de capa
  document.getElementById("toggle-insar").addEventListener("change", (e) => {
    renderInsarLayer(map, { visible: e.target.checked });
  });
  document.getElementById("insar-opacity").addEventListener("input", (e) => {
    renderInsarLayer(map, { opacity: e.target.value / 100 });
  });
}

async function loadReports() {
  const bounds = map.getBounds();
  const bbox = [
    bounds.getWest(), bounds.getSouth(),
    bounds.getEast(), bounds.getNorth(),
  ];

  try {
    const data = await getReports({ bbox });
    renderMarkers(data.reports ?? data);
    document.getElementById("last-update").textContent =
      `Actualizado: ${new Date().toLocaleTimeString("es-MX")}`;
  } catch (err) {
    console.error("Error cargando reportes:", err);
  }
}

function renderMarkers(reports) {
  markersLayer.clearLayers();
  for (const r of reports) {
    const [lng, lat] = r.location?.coordinates ?? [r.lng, r.lat];
    const color = RISK_COLOR(r.risk_index ?? 0);

    const marker = L.circleMarker([lat, lng], {
      radius: 8,
      fillColor: color,
      color: "#fff",
      weight: 1.5,
      fillOpacity: 0.9,
    });

    marker.bindTooltip(
      `<b>Riesgo ${(r.risk_index ?? 0).toFixed(1)}</b><br>${r.severity_class ?? "—"}`,
      { direction: "top" }
    );

    marker.on("click", () => openReportPanel(r.id));
    markersLayer.addLayer(marker);
  }
}
```

---

### `dashboard/src/report-panel.js`

```javascript
// report-panel.js — ficha detallada de un reporte individual

import { getReport, getSubsidencia, getRiesgo } from "./api-client.js";
import { renderTemporalChart } from "./charts.js";

export async function openReportPanel(reportId) {
  const panel = document.getElementById("report-panel");
  const content = document.getElementById("panel-content");

  content.innerHTML = "<p class='loading'>Cargando…</p>";
  panel.classList.remove("hidden");

  try {
    const [report, riesgo] = await Promise.all([
      getReport(reportId),
      getRiesgo(reportId),
    ]);

    const [lng, lat] = report.location?.coordinates ?? [0, 0];
    const insar = await getSubsidencia({ lat, lng });

    content.innerHTML = buildPanelHTML(report, riesgo, insar);

    renderTemporalChart("chart-severity", report.temporal_series ?? []);
    renderTemporalChart("chart-insar", insar.series ?? [], {
      label: "Subsidencia acumulada (mm)",
      color: "#1565c0",
    });
  } catch (err) {
    content.innerHTML = `<p class='error'>Error: ${err.message}</p>`;
  }
}

function buildPanelHTML(report, riesgo, insar) {
  const v = report.crack_metrics ?? {};
  const ri = (report.risk_index ?? 0).toFixed(2);
  const vel = (insar.velocity_mm_yr ?? 0).toFixed(1);

  return `
    <h2>Reporte #${report.id.slice(0, 8)}</h2>
    <p class="address">${report.address ?? `${report.lat?.toFixed(5)}, ${report.lng?.toFixed(5)}`}</p>

    <div class="image-pair">
      <figure>
        <img src="/api/v1/reports/${report.id}/image?type=original" alt="Foto oblicua">
        <figcaption>Foto original (oblicua)</figcaption>
      </figure>
      <figure>
        <img src="/api/v1/reports/${report.id}/image?type=ortho" alt="Ortofoto">
        <figcaption>Ortofoto corregida</figcaption>
      </figure>
    </div>

    <section class="metrics">
      <h3>Métricas de grieta</h3>
      <table>
        <tr><td>Longitud</td><td><b>${(v.length_mm ?? 0).toFixed(0)} mm</b></td></tr>
        <tr><td>Ancho medio</td><td><b>${(v.width_mean_mm ?? 0).toFixed(2)} mm</b></td></tr>
        <tr><td>Área</td><td><b>${(v.area_mm2 ?? 0).toFixed(0)} mm²</b></td></tr>
        <tr><td>Dim. fractal</td><td><b>${(v.fractal_dim ?? 0).toFixed(3)}</b></td></tr>
        <tr><td>Orientación</td><td><b>${(v.orientation_deg ?? 0).toFixed(1)}°</b></td></tr>
        <tr><td>Ramificaciones</td><td><b>${v.branch_points ?? 0}</b></td></tr>
        <tr><td>Severidad</td><td><b class="sev-${v.severity_class}">${v.severity_class ?? "—"}</b></td></tr>
      </table>
    </section>

    <section class="risk-bar">
      <h3>Índice de riesgo: <b>${ri} / 10</b></h3>
      ${renderRiskBars(riesgo)}
    </section>

    <section class="insar-summary">
      <h3>Subsidencia InSAR</h3>
      <p>Velocidad: <b>${vel} mm/año</b> |
         Aceleración: <b>${(insar.acceleration_mm_yr2 ?? 0).toFixed(1)} mm/año²</b></p>
    </section>

    <canvas id="chart-severity" height="120"></canvas>
    <canvas id="chart-insar"    height="120"></canvas>

    <div class="panel-actions">
      <button onclick="approveReport('${report.id}')" class="btn-approve">✔ Aprobar intervención</button>
      <button onclick="exportPDF('${report.id}')"     class="btn-export">📄 Exportar PDF</button>
    </div>
  `;
}

function renderRiskBars(riesgo) {
  const components = [
    { label: "Severidad (40%)", value: riesgo.severity_component ?? 0 },
    { label: "Velocidad (30%)", value: riesgo.velocity_component ?? 0 },
    { label: "Aceleración (20%)", value: riesgo.acceleration_component ?? 0 },
    { label: "Densidad (10%)", value: riesgo.density_component ?? 0 },
  ];
  return components.map(({ label, value }) => `
    <div class="risk-row">
      <span>${label}</span>
      <div class="risk-track">
        <div class="risk-fill" style="width:${(value * 10).toFixed(1)}%"></div>
      </div>
      <span>${value.toFixed(2)}</span>
    </div>
  `).join("");
}
```

---

### `dashboard/src/insar-layer.js`

```javascript
// insar-layer.js — capa raster de velocidad de subsidencia InSAR

let insarOverlay = null;

// URL del endpoint que retorna una imagen PNG/GeoTIFF coloreada del raster
const INSAR_TILE_URL = "/api/v1/subsidencia/tiles/{z}/{x}/{y}.png";

export function renderInsarLayer(map, { visible = true, opacity = 0.6 } = {}) {
  if (!insarOverlay) {
    insarOverlay = L.tileLayer(INSAR_TILE_URL, {
      opacity,
      attribution: "InSAR: Sentinel-1 / ESA",
      maxZoom: 16,
      tileSize: 256,
    });
    if (visible) insarOverlay.addTo(map);
  } else {
    insarOverlay.setOpacity(opacity);
    if (visible && !map.hasLayer(insarOverlay)) insarOverlay.addTo(map);
    if (!visible && map.hasLayer(insarOverlay)) map.removeLayer(insarOverlay);
  }
}
```

---

### `dashboard/src/priority-list.js`

```javascript
// priority-list.js — tabla paginada de reportes ordenados por risk_index DESC

import { getReports } from "./api-client.js";

const PAGE_SIZE = 50;
let currentPage = 1;
let allReports = [];

export async function initPriorityList() {
  const data = await getReports({ pageSize: 500 });   // carga inicial completa
  allReports = (data.reports ?? data).sort((a, b) => (b.risk_index ?? 0) - (a.risk_index ?? 0));
  renderPage(1);

  document.getElementById("search-priority").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    const filtered = allReports.filter(r => (r.address ?? "").toLowerCase().includes(q));
    renderRows(filtered.slice(0, PAGE_SIZE));
  });

  document.getElementById("export-csv").addEventListener("click", exportCSV);
}

function renderPage(page) {
  currentPage = page;
  const start = (page - 1) * PAGE_SIZE;
  renderRows(allReports.slice(start, start + PAGE_SIZE));
  renderPagination(allReports.length);
}

function renderRows(reports) {
  const tbody = document.getElementById("priority-tbody");
  tbody.innerHTML = reports.map((r, i) => `
    <tr class="risk-row-${riskClass(r.risk_index)}">
      <td>${i + 1}</td>
      <td>${r.address ?? `${r.lat?.toFixed(4)}, ${r.lng?.toFixed(4)}`}</td>
      <td><b>${(r.risk_index ?? 0).toFixed(2)}</b></td>
      <td>${r.severity_class ?? "—"}</td>
      <td>${(r.velocity_mm_yr ?? 0).toFixed(1)}</td>
      <td>${new Date(r.created_at).toLocaleDateString("es-MX")}</td>
      <td><button onclick="openReportPanel('${r.id}')">Ver</button></td>
    </tr>
  `).join("");
}

function renderPagination(total) {
  const pages = Math.ceil(total / PAGE_SIZE);
  const el = document.getElementById("pagination");
  el.innerHTML = Array.from({ length: pages }, (_, i) => `
    <button onclick="renderPage(${i + 1})" class="${currentPage === i + 1 ? 'active' : ''}">
      ${i + 1}
    </button>
  `).join("");
}

function riskClass(r) {
  if (r >= 8) return "high";
  if (r >= 5) return "medium-high";
  if (r >= 3) return "medium";
  return "low";
}

function exportCSV() {
  const header = "id,direccion,risk_index,severity_class,velocity_mm_yr,fecha\n";
  const rows = allReports.map(r =>
    `${r.id},"${r.address ?? ""}",${(r.risk_index ?? 0).toFixed(2)},${r.severity_class ?? ""},${(r.velocity_mm_yr ?? 0).toFixed(1)},${r.created_at}`
  ).join("\n");
  const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `grietas_toluca_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
}
```

---

### `dashboard/src/alerts.js`

```javascript
// alerts.js — alertas automáticas por riesgo crítico, subsidencia y clústeres

import { getAlerts } from "./api-client.js";

const STORAGE_KEY = "grietas_dismissed_alerts";

export async function initAlerts() {
  const dismissed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  const alerts = await getAlerts();
  const active = alerts.filter(a => !dismissed.includes(a.id));

  renderAlertBadge(active.length);
  renderAlertList(active);
}

function renderAlertBadge(count) {
  const badge = document.getElementById("alert-badge");
  badge.textContent = count > 0 ? count : "";
  badge.style.display = count > 0 ? "inline-block" : "none";
}

function renderAlertList(alerts) {
  const list = document.getElementById("alerts-list");
  if (alerts.length === 0) {
    list.innerHTML = "<li class='no-alerts'>Sin alertas activas</li>";
    return;
  }
  list.innerHTML = alerts.map(a => `
    <li class="alert alert-${a.severity}">
      <div class="alert-body">
        <b>${a.title}</b>
        <p>${a.description}</p>
        <small>${new Date(a.created_at).toLocaleString("es-MX")}</small>
      </div>
      <button onclick="dismissAlert('${a.id}')">✓ Atendida</button>
    </li>
  `).join("");
}

window.dismissAlert = function(id) {
  const dismissed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  dismissed.push(id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(dismissed));
  initAlerts();
};
```

---

### `dashboard/src/charts.js`

```javascript
// charts.js — gráficas Chart.js para series temporales

export function renderTemporalChart(canvasId, series, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !series.length) return;

  new Chart(canvas, {
    type: "line",
    data: {
      labels: series.map(p => new Date(p.date ?? p.captured_at).toLocaleDateString("es-MX")),
      datasets: [{
        label: opts.label ?? "Severity score",
        data: series.map(p => p.value ?? p.severity_score),
        borderColor: opts.color ?? "#e53935",
        backgroundColor: (opts.color ?? "#e53935") + "22",
        fill: true,
        tension: 0.3,
        pointRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } },
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}
```

---

### `dashboard/styles/dashboard.css`

```css
/* dashboard.css */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, sans-serif;
  background: #121212;
  color: #e0e0e0;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

#topbar {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 1rem;
  background: #1e1e2f;
  border-bottom: 1px solid #333;
}

.logo { font-weight: bold; font-size: 1.1rem; }

nav .tab {
  background: none;
  border: none;
  color: #aaa;
  cursor: pointer;
  padding: 0.4rem 0.8rem;
  border-radius: 4px;
}
nav .tab.active { background: #333; color: #fff; }

#last-update { margin-left: auto; font-size: 0.8rem; color: #888; }

#app { flex: 1; display: flex; overflow: hidden; }

/* Views */
.view { display: none; width: 100%; }
.view.active { display: flex; }

#view-map { position: relative; }
#map { flex: 1; height: 100%; }

/* Report panel */
#report-panel {
  width: 380px;
  background: #1e1e2f;
  border-left: 1px solid #333;
  overflow-y: auto;
  padding: 1rem;
}
#report-panel.hidden { display: none; }

#close-panel {
  float: right;
  background: none;
  border: none;
  color: #aaa;
  font-size: 1.2rem;
  cursor: pointer;
}

.image-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin: 0.5rem 0;
}
.image-pair img { width: 100%; border-radius: 4px; }
.image-pair figcaption { font-size: 0.7rem; text-align: center; color: #888; }

.metrics table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.metrics td { padding: 0.25rem 0.4rem; border-bottom: 1px solid #333; }

.sev-leve    { color: #66bb6a; }
.sev-moderado { color: #ffa726; }
.sev-severo  { color: #ef5350; }

/* Risk bars */
.risk-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.8rem; }
.risk-track { flex: 1; background: #333; border-radius: 4px; height: 8px; overflow: hidden; }
.risk-fill  { height: 100%; background: #e53935; border-radius: 4px; }

/* Layer controls */
#layer-controls {
  position: absolute;
  bottom: 2rem;
  left: 1rem;
  background: #1e1e2fcc;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  font-size: 0.85rem;
  z-index: 1000;
}

/* Priority table */
#view-priorities { flex-direction: column; padding: 1rem; }
#priority-toolbar { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
#priority-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
#priority-table th,
#priority-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #333; text-align: left; }
#priority-table th { background: #1e1e2f; }

.risk-row-high        td:nth-child(3) { color: #ef5350; }
.risk-row-medium-high td:nth-child(3) { color: #ff9800; }
.risk-row-medium      td:nth-child(3) { color: #fdd835; }

/* Alerts */
#view-alerts { flex-direction: column; padding: 1rem; }
.alert {
  background: #1e1e2f;
  border-left: 4px solid;
  border-radius: 4px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.alert-high   { border-color: #ef5350; }
.alert-medium { border-color: #ff9800; }
.alert-low    { border-color: #fdd835; }

#alert-badge {
  background: #ef5350;
  color: #fff;
  border-radius: 10px;
  padding: 0 6px;
  font-size: 0.75rem;
  margin-left: 4px;
}

.btn-approve { background: #2e7d32; }
.btn-export  { background: #1565c0; }
.panel-actions button {
  color: #fff;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 0.5rem;
  margin-right: 0.4rem;
}

.loading, .error { color: #888; padding: 1rem; }
.error { color: #ef5350; }
```

---

## Endpoints de API necesarios (Fase 5 / backend FastAPI)

Los siguientes endpoints deben existir o crearse si no están:

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/v1/reports` | GET | Lista con `bbox`, `min_risk`, `page`, `page_size` |
| `/api/v1/reports/{id}` | GET | Detalle completo + `temporal_series` |
| `/api/v1/reports/{id}/image` | GET | Imagen con `?type=original\|ortho` |
| `/api/v1/subsidencia` | GET | Timeseries InSAR para `lat,lng,radius` |
| `/api/v1/subsidencia/tiles/{z}/{x}/{y}.png` | GET | Tiles XYZ del raster de velocidad (PNG coloreado) |
| `/api/v1/riesgo/{report_id}` | GET | Componentes del índice de riesgo |
| `/api/v1/alerts` | GET | Alertas activas (riesgo ≥ 8, aceleración crítica, clústeres) |

### Tile server del raster InSAR

El endpoint `/subsidencia/tiles/{z}/{x}/{y}.png` puede implementarse con
[`rio-tiler`](https://github.com/cogeotiff/rio-tiler) o con `rasterio` +
Matplotlib colormap. Ejemplo mínimo:

```python
# api/tiles.py
import io
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from fastapi import APIRouter
from fastapi.responses import Response
import matplotlib.cm as cm

router = APIRouter()
VELOCITY_TIF = "data/insar_velocity.tif"

@router.get("/subsidencia/tiles/{z}/{x}/{y}.png")
async def insar_tile(z: int, x: int, y: int):
    """Retorna tile PNG con colormap de velocidad de subsidencia."""
    from rio_tiler.io import Reader
    with Reader(VELOCITY_TIF) as tif:
        img = tif.tile(x, y, z)
    data = img.data[0]
    # Normalizar: –50..+10 mm/año → 0..1
    norm = np.clip((data + 50) / 60, 0, 1)
    colored = (cm.RdBu_r(norm) * 255).astype(np.uint8)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(colored).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
```

---

## Despliegue

```bash
# Desarrollo local
python -m http.server 3000 --directory dashboard/

# Producción: servir como assets estáticos desde FastAPI
# api/main.py
from fastapi.staticfiles import StaticFiles
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

# Acceso: http://localhost:8000/dashboard/
```

---

## Prueba manual de aceptación

1. Abrir `http://localhost:3000` (o `/dashboard/`)
2. El mapa carga centrado en Toluca con marcadores coloreados
3. Clic en un marcador rojo → panel lateral con imágenes, métricas y gráficas
4. Tab "Prioridades" → tabla con reportes ordenados, exportable a CSV
5. Tab "Alertas" → lista de zonas críticas; clic en "Atendida" la descarta
6. Slider de opacidad cambia transparencia de la capa InSAR
7. Deschequear "Subsidencia InSAR" oculta la capa

---

## Casos borde y cómo se manejan

| Caso | Comportamiento |
|---|---|
| API no disponible | fetch catch → mensaje de error en panel, mapa vacío |
| Reporte sin ortofoto | `<img>` muestra fallback "Sin ortofoto disponible" |
| InSAR sin datos en zona | `velocity_mm_yr = 0`, gráfica con serie vacía |
| Sin alertas activas | Texto "Sin alertas activas" en la vista |
| Mapa sin reportes en bbox | `markersLayer` vacío, no crash |
| Token JWT expirado | `apiFetch` lanza Error; añadir interceptor para redirigir a `/login` |

---

## Siguiente paso

Con la Fase 9 completa, el sistema cubre el flujo end-to-end:

```
Ciudadano captura foto (PWA, Fase 4)
  → Backend recibe y clasifica (Fases 5, 3)
  → Ortorectifica (Fase 6) y segmenta (Fase 7)
  → Calcula riesgo con InSAR (Fase 8)
  → Municipal visualiza y prioriza (Fase 9)
```

El siguiente ciclo de mejora recomendado:
- Autenticación municipal con OAuth2/JWT
- Exportación de informe PDF por zona (jsPDF)
- Integración con sistema de órdenes de trabajo municipal existente
- Modelo de predicción: regresión de evolución de severidad para los próximos 6 meses
