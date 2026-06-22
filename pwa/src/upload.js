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
        await saveToOfflineQueue(imageBlob, metadata);
        throw new Error("Sin conexión. Reporte guardado para envío posterior.");
      }
      await new Promise(r => setTimeout(r, 2 ** attempt * 1000));
    }
  }
}

/** Registra un intento de sincronización cuando se recupere la red. */
export async function registerSyncOnReconnect() {
  if ("serviceWorker" in navigator && "SyncManager" in window) {
    const reg = await navigator.serviceWorker.ready;
    await reg.sync.register("sync-reports");
  }
}

export async function saveToOfflineQueue(imageBlob, metadata) {
  const db  = await openDB();
  const id  = crypto.randomUUID();
  const tx  = db.transaction("queue", "readwrite");
  tx.objectStore("queue").put({
    id,
    imageBlob,
    metadata,
    savedAt: new Date().toISOString(),
  });
  await new Promise((res, rej) => {
    tx.oncomplete = res;
    tx.onerror    = e => rej(e.target.error);
  });
  await registerSyncOnReconnect();
}

/** Retorna todos los reportes pendientes de la cola offline. */
export async function getOfflineQueue() {
  const db    = await openDB();
  const tx    = db.transaction("queue", "readonly");
  const store = tx.objectStore("queue");
  return new Promise((resolve, reject) => {
    const req      = store.getAll();
    req.onsuccess  = e => resolve(e.target.result);
    req.onerror    = e => reject(e.target.error);
  });
}

/** Elimina un reporte de la cola tras enviarlo con éxito. */
export async function removeFromOfflineQueue(id) {
  const db  = await openDB();
  const tx  = db.transaction("queue", "readwrite");
  tx.objectStore("queue").delete(id);
  return new Promise((res, rej) => {
    tx.oncomplete = res;
    tx.onerror    = e => rej(e.target.error);
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
