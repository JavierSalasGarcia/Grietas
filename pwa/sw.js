const CACHE_NAME = "grietas-pwa-v1";
const STATIC_ASSETS = [
  "/",
  "/index.html",
  "/styles/main.css",
  "/src/camera.js",
  "/src/orientation.js",
  "/src/geolocation.js",
  "/src/ua-camera-db.js",
  "/src/citizen-store.js",
  "/src/calibration.js",
  "/src/upload.js",
  "/manifest.json",
];

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
  if (url.pathname.startsWith("/api/")) return;
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
  const dbReq = indexedDB.open("grietas-offline", 1);
  const db = await new Promise((resolve, reject) => {
    dbReq.onsuccess = e => resolve(e.target.result);
    dbReq.onerror   = e => reject(e.target.error);
  });

  const tx      = db.transaction("queue", "readonly");
  const store   = tx.objectStore("queue");
  const allReq  = store.getAll();
  const records = await new Promise((resolve, reject) => {
    allReq.onsuccess = e => resolve(e.target.result);
    allReq.onerror   = e => reject(e.target.error);
  });

  for (const record of records) {
    try {
      const form = new FormData();
      form.append("image",    record.imageBlob, `report_${record.id}.jpg`);
      form.append("metadata", JSON.stringify(record.metadata));
      const resp = await fetch("/api/v1/reports", { method: "POST", body: form });
      if (resp.ok) {
        const delTx = db.transaction("queue", "readwrite");
        delTx.objectStore("queue").delete(record.id);
      }
    } catch (_) {
      // Dejar en cola para el próximo intento de sync
    }
  }
}
