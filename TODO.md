# TODO — Sistema de Reporte de Grietas en Calles de Toluca

## Descripción del sistema

Plataforma ciudadana para reportar grietas y fisuras en el pavimento de Toluca.
Los ciudadanos fotografían las grietas mediante una PWA. Un modelo de IA clasifica
cada reporte como **Aprobado** (grieta real) o **No Aprobado** (foto inválida).
Los reportes aprobados se ortorectifican, se mide la grieta en milímetros y se
correlaciona con datos de subsidencia del suelo obtenidos del satélite Sentinel-1
para priorizar el mantenimiento municipal.

---

## Problema técnico documentado: pérdida de EXIF en PWA con canvas overlay

### Naturaleza del problema

La app es una **PWA** (Progressive Web App) que muestra un overlay sobre la cámara:
la última fotografía del mismo punto en transparencia, para que el ciudadano pueda
reproducir el mismo ángulo de captura en visitas sucesivas. Este overlay se
renderiza mediante un elemento `<canvas>` superpuesto al stream de `getUserMedia()`.

Cuando el usuario presiona el shutter, la imagen se captura con `canvas.toBlob()`
o `canvas.toDataURL()`. El canvas genera píxeles nuevos y **descarta
completamente la cadena de metadatos EXIF** que el sensor de la cámara embebe
en el JPEG original. Esto ocurre en todos los navegadores y es intrínseco al
modelo de seguridad del canvas, no un bug.

### Datos que se pierden

| Dato EXIF perdido | Impacto en el sistema |
|---|---|
| `Orientation` / giroscopio | No se puede construir la matriz de rotación R para ortorectificación |
| `FocalLength`, `FocalLengthIn35mmFilm` | No se puede construir la matriz intrínseca K de la cámara |
| `GPSLatitude`, `GPSLongitude` | La georreferenciación no queda en el archivo |
| `DateTimeOriginal` | El timestamp queda solo en el nombre del archivo si no se gestiona |
| `Make`, `Model` | No se puede consultar la base de datos de specs de cámara |

### Solución adoptada: sidecar JSON de metadatos

La PWA **no depende del EXIF**. En el momento exacto del shutter, el JavaScript
de la app captura todos los datos necesarios desde las APIs del navegador y los
envía como un JSON independiente junto con la imagen:

```
POST /api/reports  (multipart/form-data)
  ├── image     → JPEG sin EXIF  (del canvas)
  └── metadata  → JSON con orientación, GPS, altura ciudadano, specs de cámara
```

### Fuentes de datos según disponibilidad del navegador

| Dato necesario | API principal | Fallback |
|---|---|---|
| Pitch / Roll / Yaw | `deviceorientationabsolute` event | `deviceorientation` event |
| GPS | `navigator.geolocation.getCurrentPosition()` | — |
| Timestamp | `Date.now()` en JS | — |
| Intrínsecos de cámara (K) | WebXR `XRCamera.projectionMatrix` (Chrome Android) | UA database → prior fijo |
| Resolución del sensor | `MediaTrackSettings` vía `track.getSettings()` | EXIF si no hay canvas |
| Modelo del dispositivo | `navigator.userAgent` | — |

### Confiabilidad de la ortorectificación por fuente de intrínsecos

El campo `intrinsics_source` del JSON registra cómo se estimó K:

| Valor | Fuente | Error GSD esperado |
|---|---|---|
| `"webxr"` | WebXR `XRCamera` (Chrome+ARCore) | < 2 % |
| `"onboarding_calibration"` | Foto de cuadrícula en primer uso | < 5 % |
| `"ua_database"` | Base de datos de specs por modelo | < 12 % |
| `"prior"` | FoV fijo 70° asumido | < 18 % |

### Limitación en iOS Safari

`getUserMedia()` + canvas en iOS Safari también elimina EXIF. `deviceorientation`
funciona pero requiere permiso explícito del usuario desde iOS 13
(`DeviceOrientationEvent.requestPermission()`). WebXR no está disponible en
Safari. Por tanto en iOS siempre se usa `ua_database` o `prior` para los
intrínsecos.

---

## Estado actual del repositorio

```
Rama activa:  claude/pavement-crack-downloader-ywz2ai
Repositorio:  javiersalasgarcia/grietas

Archivos presentes:
  download_crack_datasets.py  ← Fase 0 completada
  contexto.md                 ← Descripción general del proyecto
  TODO.md                     ← Este archivo
  .gitignore                  ← Excluye pavement_crack_datasets/
```

---

## Decisiones de diseño fijadas (no reabrir)

| Decisión | Valor fijado | Razón |
|---|---|---|
| Ángulo de captura recomendado | 45° desde vertical | Balance entre cobertura de área y resolución de ancho de grieta |
| Altura estimada de cámara | `citizen_height_cm × 0.85 / 100` metros | Teléfono sostenido a altura de hombro/cabeza |
| GSD objetivo | 0.5–1.0 mm/px | Permite medir grietas ≥ 1.5 mm de ancho |
| Temperatura de destilación | T = 3 | Estándar para tareas de clasificación binaria |
| Peso de pérdida mixta | α = 0.5 | Balance igual entre CE dura y KL suave |
| Imágenes COCO "No Aprobado" | val2017 completo (5 000 imgs) | Exactamente 5 000, sin categorías de pavimento dañado |
| Semilla aleatoria COCO | 42 | Reproducibilidad |
| Formato de anotación de grietas | YOLOv8 (Roboflow) | Ecosistema Ultralytics, permite detección futura |
| Resolución Sentinel-1 | 15 m/píxel, cada 12 días | Parámetros fijos del satélite, no modificables |
| Ventana temporal InSAR | 2014–2026 (en curso) | Desde primera imagen disponible de Toluca |

---

## Fases de desarrollo

---

### FASE 0 — Construcción del dataset  `[COMPLETADA ✅]`

**Objetivo**: Descargar imágenes de grietas (Aprobado) e imágenes cotidianas
(No Aprobado) de fuentes públicas y generar un registro consolidado.

**Archivo**: `download_crack_datasets.py`

**Fuentes descargadas**:
1. Roboflow Universe `mcmaster-vkq5k/pavement-crack-detection-r4n7n` → requiere `ROBOFLOW_API_KEY`
2. Mendeley Data dataset `88kdyyc73h` v1 → API pública, sin auth
3. GitHub `juhuyan/CrackDataset_DL_HY` → API pública, `GITHUB_TOKEN` opcional
4. Figshare artículo `25103138` → API pública, sin auth
5. COCO val2017 → descarga directa desde `images.cocodataset.org`

**Salida**:
```
pavement_crack_datasets/
  01_roboflow/extracted/   ← imágenes + labels YOLOv8
  02_mendeley/             ← archivos del dataset
  03_github/               ← estructura del repo
  04_figshare/             ← archivos extraídos
  05_coco_control/
    images/                ← JPEGs con prefijo coco_
    annotations/           ← JSON por imagen con categorías y bboxes COCO
  download_registry.csv    ← índice con campos: id, source, filename,
                              local_path, source_url, file_type,
                              categoria_proyecto, label, annotation_file,
                              md5, size_bytes, downloaded_at
  download_registry.json   ← mismo índice en JSON
  download.log
```

**Ejecución**:
```bash
pip install requests
export ROBOFLOW_API_KEY=<clave>
export GITHUB_TOKEN=<token>   # opcional
python download_crack_datasets.py
```

---

### FASE 1 — Preprocesamiento y validación del dataset  `[PENDIENTE]`

**Objetivo**: Limpiar, validar, balancear y dividir el dataset para entrenamiento.

**Entrada**: `pavement_crack_datasets/download_registry.csv`

**Archivo a crear**: `preprocess_dataset.py`

**Pasos a implementar**:

1. **Leer el registro** con `pandas`:
   ```python
   df = pd.read_csv("pavement_crack_datasets/download_registry.csv")
   images = df[df["file_type"] == "image"]
   ```

2. **Verificar integridad** de cada imagen:
   ```python
   from PIL import Image
   def is_valid(path):
       try:
           Image.open(path).verify()
           return True
       except Exception:
           return False
   ```
   Registrar archivos corruptos en `corrupt_files.txt` y eliminarlos del dataset.

3. **Redimensionar** todas las imágenes a `224 × 224` px (requerido por DINOv2
   y MobileNetV3). Guardar en `dataset_processed/` manteniendo la estructura
   `Aprobado/` y `No_Aprobado/` (formato ImageFolder de PyTorch).

4. **Balancear clases**: contar imágenes por categoría. Si el desbalance es
   mayor que 3:1, aplicar submuestreo aleatorio de la clase mayoritaria con
   semilla fija (seed=42).

5. **Dividir en splits** (70 / 15 / 15):
   ```
   dataset_processed/
     train/
       Aprobado/
       No_Aprobado/
     val/
       Aprobado/
       No_Aprobado/
     test/
       Aprobado/
       No_Aprobado/
   ```
   Guardar los splits como `split_train.csv`, `split_val.csv`, `split_test.csv`
   con columnas `[local_path, categoria_proyecto, source]`.

6. **Calcular estadísticas** del dataset:
   - Total imágenes por split y por categoría
   - Media y desviación estándar de RGB (para normalización)
   - Guardar en `dataset_stats.json`

**Salida esperada**:
```
dataset_processed/
  train/Aprobado/      (~2 100 imgs si dataset balanceado en 3 000 total)
  train/No_Aprobado/   (~2 100 imgs)
  val/Aprobado/        (~450 imgs)
  val/No_Aprobado/     (~450 imgs)
  test/Aprobado/       (~450 imgs)
  test/No_Aprobado/    (~450 imgs)
  split_train.csv
  split_val.csv
  split_test.csv
  dataset_stats.json   ← {mean_rgb, std_rgb, n_train, n_val, n_test, ...}
```

**Dependencias**: `pip install pandas Pillow tqdm`

---

### FASE 2 — Entrenamiento del modelo Teacher (DINOv2)  `[PENDIENTE]`

**Objetivo**: Fine-tuning de DINOv2-Base como clasificador binario de alta precisión.
Este modelo no se despliega en producción; su función es generar soft labels
de alta calidad para la Fase 3.

**Archivo a crear**: `train_teacher.py`

**Arquitectura**:
```
DINOv2-Base (ViT-B/14, 86 M params, preentrenado en LVD-142M)
    ↓  [congelar primeras 10 capas de los 12 bloques]
    ↓  [fine-tune últimas 2 capas + cabeza]
Linear head: 768 → 512 → 2
Loss: CrossEntropyLoss con class_weights para desbalance residual
```

**Referencia del modelo**: `facebookresearch/dinov2` vía `torch.hub` o
`from transformers import AutoModel` (HuggingFace `facebook/dinov2-base`).

**Hiperparámetros de partida** (ajustar con búsqueda si accuracy < 92 %):
```python
BATCH_SIZE    = 32
LR_BACKBONE   = 1e-5   # muy bajo para no destruir features preentrenados
LR_HEAD       = 1e-3
EPOCHS        = 20
WEIGHT_DECAY  = 1e-4
SCHEDULER     = CosineAnnealingLR(T_max=20)
```

**Augmentation de entrenamiento** (aplicar solo a `train/`):
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean=dataset_stats["mean_rgb"],
                         std=dataset_stats["std_rgb"]),
])
```

**Métricas a reportar en cada epoch**:
- `accuracy`, `precision`, `recall`, `f1` sobre `val/`
- Guardar checkpoint si `val_f1 > mejor_anterior`

**Criterio de aceptación**: `val_accuracy ≥ 95 %` y `val_recall_Aprobado ≥ 94 %`
(priorizar no perder grietas reales → recall alto en la clase positiva).

**Salida**:
```
models/
  teacher_dinov2_best.pth      ← pesos del mejor checkpoint
  teacher_dinov2_config.json   ← hiperparámetros y métricas finales
  teacher_confusion_matrix.png
```

**Dependencias**: `pip install torch torchvision transformers scikit-learn matplotlib`

---

### FASE 3 — Destilación del modelo Student (MobileNetV3)  `[PENDIENTE]`

**Objetivo**: Entrenar un modelo ligero (< 10 MB, < 200 ms en CPU) destilando
el conocimiento del Teacher de la Fase 2.

**Archivo a crear**: `train_student.py`

**Arquitectura Student**:
```
MobileNetV3-Small (2.5 M params, torchvision.models.mobilenet_v3_small)
  → modificar classifier[-1]: Linear(576, 2)
```

**Función de pérdida de destilación**:
```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, hard_labels,
                      T=3, alpha=0.5):
    # Pérdida suave: aprende la distribución de confianza del teacher
    soft_teacher  = F.softmax(teacher_logits / T, dim=1)
    soft_student  = F.log_softmax(student_logits / T, dim=1)
    loss_soft     = F.kl_div(soft_student, soft_teacher,
                             reduction='batchmean') * (T ** 2)
    # Pérdida dura: aprende las etiquetas ground-truth
    loss_hard     = F.cross_entropy(student_logits, hard_labels)
    return alpha * loss_hard + (1 - alpha) * loss_soft
```

**Proceso de destilación**:
1. Cargar Teacher (Fase 2) en modo `eval()`, sin gradientes
2. Por cada batch: pasar por Teacher para obtener `teacher_logits`
3. Calcular `distillation_loss(student_logits, teacher_logits, labels)`
4. Backprop solo sobre el Student

**Hiperparámetros**:
```python
T           = 3      # temperatura de destilación
ALPHA       = 0.5    # peso pérdida dura vs. suave
LR          = 1e-3
EPOCHS      = 30
BATCH_SIZE  = 64
```

**Criterio de aceptación**:
- `val_accuracy ≥ 90 %`
- Tamaño del archivo `.onnx` ≤ 15 MB
- Inferencia ≤ 200 ms en CPU (medir con `timeit` en imagen 224×224)

**Exportar a ONNX** para despliegue:
```python
torch.onnx.export(
    student_model,
    dummy_input,                    # tensor (1, 3, 224, 224)
    "models/student_mobilenetv3.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
    opset_version=17,
)
```

**Salida**:
```
models/
  student_mobilenetv3_best.pth
  student_mobilenetv3.onnx        ← modelo para producción
  student_config.json             ← métricas, hiperparámetros, benchmark
  student_confusion_matrix.png
```

---

### FASE 4 — PWA Frontend: captura de imagen y metadatos  `[PENDIENTE]`

**Objetivo**: Construir la interfaz de captura ciudadana. No depende de EXIF.
Todos los metadatos se capturan explícitamente en JavaScript.

**Directorio a crear**: `pwa/`

**Tecnología**: HTML5 + Vanilla JS (sin frameworks pesados para PWA ligera).
Alternativa aceptable: Vue 3 + Vite si el equipo prefiere componentes.

**Flujo de la cámara**:
```javascript
// 1. Stream de la cámara trasera
const stream = await navigator.mediaDevices.getUserMedia({
  video: { facingMode: 'environment', width: 4000, height: 3000 }
});
videoElement.srcObject = stream;

// 2. Leer specs del stream activo
const track    = stream.getVideoTracks()[0];
const settings = track.getSettings();  // width, height, deviceId, frameRate

// 3. Overlay: última foto del punto en transparencia (alpha = 0.35)
// Dibujar en canvas: primero frame del video, encima la foto anterior
ctx.drawImage(videoElement, 0, 0);
ctx.globalAlpha = 0.35;
ctx.drawImage(previousPhotoElement, 0, 0);
ctx.globalAlpha = 1.0;
```

**Captura de orientación** (registrar continuamente, tomar snapshot al shutter):
```javascript
let _orient = { alpha: null, beta: null, gamma: null, absolute: false };

// Preferir orientación absoluta (relativa al Norte magnético)
window.addEventListener('deviceorientationabsolute', e => {
  _orient = { alpha: e.alpha, beta: e.beta, gamma: e.gamma, absolute: true };
}, true);

// Fallback: orientación relativa
window.addEventListener('deviceorientation', e => {
  if (!_orient.absolute)
    _orient = { alpha: e.alpha, beta: e.beta, gamma: e.gamma, absolute: false };
});

// En iOS 13+ se requiere permiso explícito:
async function requestOrientationPermission() {
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    const perm = await DeviceOrientationEvent.requestPermission();
    if (perm !== 'granted') throw new Error('Permiso de giroscopio denegado');
  }
}
```

**Función de captura** (shutter):
```javascript
async function capture() {
  const ts = Date.now();

  // Capturar frame limpio del video (sin overlay) para análisis
  const cleanCtx = cleanCanvas.getContext('2d');
  cleanCtx.drawImage(videoElement, 0, 0);

  const [imageBlob, gpsPosition] = await Promise.all([
    new Promise(res => cleanCanvas.toBlob(res, 'image/jpeg', 0.92)),
    new Promise((res, rej) =>
      navigator.geolocation.getCurrentPosition(res, rej, { enableHighAccuracy: true })
    ).catch(() => null),
  ]);

  const metadata = {
    schema_version: "1.1",
    captured_at:    new Date(ts).toISOString(),
    citizen: {
      height_cm:  citizenProfile.heightCm,    // registrado en onboarding
      device_id:  await getDeviceFingerprint(),
      user_agent: navigator.userAgent,
    },
    camera: {
      orientation:       { ..._orient },       // snapshot del giroscopio
      resolution:        { width: settings.width, height: settings.height },
      stream_settings:   settings,
      fov_estimate_deg:  estimateFovFromUA(navigator.userAgent),
      intrinsics_source: "ua_database",        // actualizar si WebXR disponible
    },
    gps: gpsPosition ? {
      lat:        gpsPosition.coords.latitude,
      lng:        gpsPosition.coords.longitude,
      alt_m:      gpsPosition.coords.altitude,
      accuracy_m: gpsPosition.coords.accuracy,
    } : null,
    previous_report_id: currentOverlayReportId ?? null,
    capture_conditions: {
      overlay_used: currentOverlayReportId !== null,
    },
  };

  // Subir al backend
  const form = new FormData();
  form.append('image',    imageBlob, `report_${ts}.jpg`);
  form.append('metadata', JSON.stringify(metadata));
  const resp = await fetch('/api/v1/reports', { method: 'POST', body: form });
  return resp.json();  // { report_id, status }
}
```

**Onboarding de calibración** (primer uso del dispositivo):
- Mostrar pantalla con cuadrícula de 10×10 cm impresa o en pantalla secundaria
- El usuario coloca el teléfono plano sobre ella y fotografía
- El servidor estima K y lo almacena asociado al `device_id`
- Actualiza `intrinsics_source` a `"onboarding_calibration"` en futuros reportes

**Archivos del módulo PWA**:
```
pwa/
  index.html
  camera.js          ← stream, overlay, captura
  orientation.js     ← DeviceOrientationEvent listener
  geolocation.js     ← GPS con timeout y fallback
  ua-camera-db.js    ← tabla FoV por modelo (top 200 dispositivos en México)
  calibration.js     ← flujo de calibración en onboarding
  upload.js          ← multipart POST con retry
  manifest.json      ← config PWA (icon, theme, display: standalone)
  sw.js              ← Service Worker: offline queue de reportes no subidos
```

**Service Worker — cola offline**:
Los reportes capturados sin conexión se guardan en `IndexedDB` y se suben
automáticamente cuando se recupera la conectividad.

---

### FASE 5 — Backend API (FastAPI)  `[PENDIENTE]`

**Objetivo**: Recibir reportes de la PWA, orquestar el pipeline de IA y
persistir los resultados.

**Directorio a crear**: `api/`

**Tecnología**: FastAPI + Uvicorn. Base de datos: PostgreSQL con PostGIS
(necesario para consultas espaciales con los datos de subsidencia).

**Endpoints**:

```
POST   /api/v1/reports
       Recibe: multipart (image + metadata JSON)
       Responde: { report_id, status: "queued" }
       Acción: guarda imagen y metadata, encola tarea de clasificación

GET    /api/v1/reports/{report_id}
       Responde: { report_id, categoria, confianza, grieta_metricas, created_at }

GET    /api/v1/reports?lat=&lng=&radio_m=&desde=&hasta=
       Responde: lista de reportes en área y rango de fechas

GET    /api/v1/subsidencia?lat=&lng=
       Responde: serie temporal de subsidencia en el punto más cercano
                 { fechas: [...], valores_mm: [...], velocidad_mm_anio: float }

GET    /api/v1/riesgo?lat=&lng=&radio_m=
       Responde: índice de riesgo integrado R(x,y,t) para la zona
```

**Pipeline de procesamiento** (tarea asíncrona con Celery o BackgroundTasks):
```
Imagen recibida
    ↓
[1] Clasificación (Fase 3 student ONNX)
    → si "No Aprobado": marcar y terminar
    ↓ si "Aprobado"
[2] Ortorectificación (Fase 6)
    → ortofoto con GSD conocido
    ↓
[3] Segmentación de grieta (Fase 7)
    → máscara binaria
    ↓
[4] Extracción métrica (Fase 7)
    → v_grieta = [w_mm, L_mm, A_mm2, D_f, theta_N]
    ↓
[5] Consulta subsidencia en (lat, lng) (Fase 8)
    → s, ds/dt, d2s/dt2
    ↓
[6] Cálculo índice de riesgo R(x,y,t)
    → persistir en PostgreSQL + PostGIS
```

**Modelo de datos (tabla `reports`):**
```sql
CREATE TABLE reports (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  image_path      TEXT NOT NULL,
  metadata_json   JSONB NOT NULL,
  categoria       TEXT CHECK (categoria IN ('Aprobado','No Aprobado','Pendiente')),
  confianza       FLOAT,
  location        GEOMETRY(Point, 4326),   -- PostGIS
  -- Métricas de grieta (NULL si No Aprobado)
  crack_width_mm  FLOAT,
  crack_length_mm FLOAT,
  crack_area_mm2  FLOAT,
  crack_fractal   FLOAT,
  crack_angle_deg FLOAT,
  -- Subsidencia en el punto al momento del reporte
  subsid_mm       FLOAT,
  subsid_vel_mm_yr FLOAT,
  subsid_accel    FLOAT,
  -- Índice de riesgo calculado
  risk_index      FLOAT,
  -- Enlace serie temporal
  previous_report_id UUID REFERENCES reports(id),
  -- Intrínsecos usados (para auditoría)
  intrinsics_source TEXT
);
```

**Dependencias**:
```
pip install fastapi uvicorn[standard] onnxruntime sqlalchemy
pip install asyncpg psycopg2-binary geoalchemy2 celery redis
pip install opencv-python-headless numpy pillow
```

---

### FASE 6 — Módulo de ortorectificación  `[PENDIENTE]`

**Objetivo**: Transformar la foto oblicua a 45° en una ortofoto métrica
(vista cenital con escala en mm/px) usando los datos del JSON sidecar.

**Archivo a crear**: `api/orthorectify.py`

**Entrada**: imagen PIL/numpy + diccionario `metadata["camera"]` + `metadata["citizen"]`

**Salida**: imagen ortorectificada numpy array + GSD en mm/px

**Implementación**:

```python
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def build_K(fov_h_deg: float, width_px: int, height_px: int) -> np.ndarray:
    """Matriz intrínseca de la cámara desde FoV horizontal."""
    fov_h_rad = np.radians(fov_h_deg)
    fx = (width_px / 2) / np.tan(fov_h_rad / 2)
    fy = fx * (height_px / width_px)   # asume píxeles cuadrados
    cx, cy = width_px / 2, height_px / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def build_R(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """
    Rotación desde ángulos de DeviceOrientationEvent.
    alpha = yaw (azimut relativo al Norte), beta = pitch, gamma = roll
    Convención: Z-X-Y (yaw → pitch → roll)
    """
    R = Rotation.from_euler('zxy',
                            [alpha_deg, beta_deg, gamma_deg],
                            degrees=True)
    return R.as_matrix()

def compute_gsd(h_m: float, fov_h_deg: float,
                pitch_deg: float, width_px: int) -> float:
    """Ground Sampling Distance en mm/px."""
    slant = h_m / max(np.cos(np.radians(pitch_deg)), 0.01)
    ground_width_m = 2 * slant * np.tan(np.radians(fov_h_deg / 2))
    return (ground_width_m * 1000) / width_px   # mm/px

def orthorectify(image: np.ndarray, metadata: dict) -> tuple[np.ndarray, float]:
    """
    Retorna (ortofoto, gsd_mm_per_px).
    """
    cam   = metadata["camera"]
    h_m   = metadata["citizen"]["height_cm"] / 100 * 0.85
    orient = cam["orientation"]

    W, H   = cam["resolution"]["width"], cam["resolution"]["height"]
    fov_h  = cam.get("fov_estimate_deg", 70.0)
    pitch  = orient.get("beta", 45.0)

    K = build_K(fov_h, W, H)
    R = build_R(orient.get("alpha", 0),
                orient.get("beta",  45),
                orient.get("gamma", 0))

    # Homografía: proyección del plano del suelo (z=0) a la imagen
    t = np.array([0.0, 0.0, -h_m])
    H_mat = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H_inv = np.linalg.inv(H_mat)

    # Tamaño de salida: aproximadamente el footprint proyectado
    gsd = compute_gsd(h_m, fov_h, pitch, W)
    out_w = int((W * gsd) / 1.0)   # mantener escala 1:1 mm/px sería enorme;
    out_h = int((H * gsd) / 1.0)   # ajustar a resolución de salida deseada
    output_size = (min(out_w, 2048), min(out_h, 2048))

    ortho = cv2.warpPerspective(image, H_inv, output_size,
                                flags=cv2.INTER_LANCZOS4)
    return ortho, gsd
```

**Casos especiales a manejar**:
- `beta` (pitch) fuera del rango [30°, 60°]: rechazar ortorectificación,
  marcar `ortho_quality = "poor"`, usar imagen original para segmentación
- `intrinsics_source == "prior"`: añadir factor de incertidumbre ±18% al GSD
  reportado, usar intervalo en lugar de valor puntual para métricas

---

### FASE 7 — Segmentación de grieta e índice de severidad  `[PENDIENTE]`

**Objetivo**: A partir de la ortofoto (Fase 6), producir la máscara de la grieta
y calcular el vector de severidad métrico `v_grieta`.

**Archivo a crear**: `api/crack_analysis.py`

**Modelo de segmentación recomendado**: DeepCrack (Liu et al., 2019)
- Repositorio: `https://github.com/yhlleo/DeepCrack`
- Input: imagen 512×512 RGB
- Output: máscara binaria de grieta

Alternativa más moderna: **CrackFormer** (Liu et al., 2021) — mejor en grietas finas.

**Pipeline de análisis**:

```python
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def analyze_crack(ortho: np.ndarray, gsd_mm_px: float) -> dict:
    """
    ortho:      imagen ortorectificada (numpy BGR o RGB)
    gsd_mm_px:  milímetros por píxel

    Retorna vector de severidad con todas las medidas en mm o mm².
    """
    # 1. Segmentación (pasar por modelo DeepCrack)
    mask = run_deepcrack(ortho)    # ndarray binario uint8

    # 2. Esqueleto para longitud y topología
    skeleton  = skeletonize(mask > 0)
    n_pixels  = np.sum(mask > 0)
    skel_px   = np.sum(skeleton)

    # 3. Ancho medio: área / longitud del esqueleto
    L_mm   = skel_px  * gsd_mm_px
    A_mm2  = n_pixels * (gsd_mm_px ** 2)
    w_mean = A_mm2 / L_mm if L_mm > 0 else 0.0

    # 4. Dimensión fractal por box-counting
    D_f = box_counting_dimension(mask)

    # 5. Orientación dominante (histograma de gradientes de la máscara)
    theta_N = dominant_orientation(mask, north_offset_deg=0)

    # 6. Número de bifurcaciones (complejidad topológica)
    n_branches = count_branch_points(skeleton)

    # 7. Fracción del área dañada
    A_ratio = n_pixels / (ortho.shape[0] * ortho.shape[1])

    return {
        "crack_width_mm":    round(w_mean, 2),
        "crack_length_mm":   round(L_mm, 1),
        "crack_area_mm2":    round(A_mm2, 1),
        "crack_fractal_dim": round(D_f, 3),
        "crack_angle_deg":   round(theta_N, 1),
        "crack_area_ratio":  round(A_ratio, 4),
        "crack_branches":    n_branches,
        "severity_class":    classify_severity(w_mean, A_ratio),
        "gsd_mm_px":         round(gsd_mm_px, 3),
    }

def classify_severity(w_mm: float, a_ratio: float) -> str:
    """Clasificación por ASTM D6433 simplificada."""
    if w_mm < 0.3 or a_ratio < 0.005:
        return "leve"
    elif w_mm < 3.0 and a_ratio < 0.05:
        return "moderado"
    else:
        return "severo"
```

**Evolución temporal** (si existe `previous_report_id`):
```python
def compute_evolution(current: dict, previous: dict, delta_days: float) -> dict:
    return {
        "delta_width_mm":      current["crack_width_mm"]  - previous["crack_width_mm"],
        "delta_length_mm":     current["crack_length_mm"] - previous["crack_length_mm"],
        "growth_rate_mm_month": (current["crack_length_mm"] - previous["crack_length_mm"])
                                / (delta_days / 30),
        "accelerating":        None,   # requiere ≥ 3 mediciones
    }
```

---

### FASE 8 — Integración de datos InSAR de subsidencia  `[PENDIENTE]`

**Objetivo**: Ingestar y consultar los datos de subsidencia Sentinel-1 y calcular
el índice de riesgo integrado R(x,y,t).

**Características de los datos**:
- Satélite: Sentinel-1 (banda C, λ = 5.6 cm)
- Resolución espacial: 15 m/píxel
- Resolución temporal: 12 días (una imagen cada 12 días)
- Cobertura temporal: 2014 – presente (se actualiza con cada nueva adquisición)
- Procesamiento: PS-InSAR o SBAS con SNAP + StaMPS o MintPy
- Producto: mapas de desplazamiento acumulado en mm (LOS — Line Of Sight)

**Formato de entrada esperado** (por confirmar con el equipo de procesamiento InSAR):
```
insar_data/
  toluca_displacement_YYYYMMDD.tif   ← GeoTIFF, una banda, mm de desplazamiento
  toluca_velocity.tif                ← mm/año promedio del período completo
  metadata.json                      ← { epsg, resolution_m, bbox, dates: [...] }
```

**Módulo a crear**: `api/insar.py`

**Ingesta** (ejecutar cada vez que llega nueva imagen del satélite):
```python
import rasterio
import numpy as np
from datetime import date

def ingest_insar_image(tif_path: str, acquisition_date: date, db_session):
    """
    Lee el GeoTIFF de desplazamiento y escribe en la tabla
    insar_displacement(date, x_pixel, y_pixel, displacement_mm, lat, lng).
    Para datasets grandes usar bulk insert con COPY.
    """
    with rasterio.open(tif_path) as src:
        data      = src.read(1)           # banda única en mm
        transform = src.transform
        crs       = src.crs
    # Convertir a puntos + PostGIS para consultas espaciales eficientes
    ...
```

**Consulta por punto GPS**:
```python
def get_subsidence_timeseries(lat: float, lng: float, db) -> dict:
    """
    Retorna la serie temporal de subsidencia para el píxel más cercano.
    También calcula velocidad (mm/año) y aceleración (mm/año²).
    """
    rows = db.execute("""
        SELECT date, displacement_mm
        FROM insar_displacement
        ORDER BY location <-> ST_SetSRID(ST_Point(:lng, :lat), 4326)
        LIMIT 1 PER GROUP ...   -- una fila por fecha, píxel más cercano
    """, {"lat": lat, "lng": lng}).fetchall()

    dates  = [r.date for r in rows]
    values = np.array([r.displacement_mm for r in rows])
    vel    = np.polyfit(range(len(values)), values, 1)[0] * 365/12  # mm/año
    accel  = np.polyfit(range(len(values)), np.diff(values, prepend=values[0]), 1)[0]

    return {
        "dates":          [d.isoformat() for d in dates],
        "displacement_mm": values.tolist(),
        "velocity_mm_yr":  round(vel, 2),
        "acceleration":    round(accel, 4),
    }
```

**Índice de riesgo integrado**:
```
R(x,y,t) = α · severity_score(v_grieta)
          + β · |ṡ(x,y,t)|          ← velocidad de subsidencia (mm/12días)
          + γ · max(s̈(x,y,t), 0)   ← aceleración positiva (hundiéndose más rápido)
          + δ · ρ(x,y,t)            ← densidad de reportes en radio de 50 m

Valores iniciales (ajustar con datos reales):
  α = 0.40,  β = 0.30,  γ = 0.20,  δ = 0.10
  R ∈ [0, 10]

severity_score:
  leve=2, moderado=5, severo=9  (de classify_severity())
```

**Dependencias**: `pip install rasterio geopandas shapely`

---

### FASE 9 — Dashboard municipal  `[PENDIENTE]`

**Objetivo**: Visualización geoespacial del estado de las grietas y subsidencia
para el equipo municipal de mantenimiento de vialidades.

**Tecnología sugerida**: Leaflet.js + Chart.js (ligero, sin dependencias pesadas).
Alternativa: Deck.gl para visualización de tensores spatio-temporales.

**Vistas principales**:

1. **Mapa de calor de riesgo**: capa de riesgo R(x,y) superpuesta al mapa de Toluca,
   coloreada por semáforo (verde → amarillo → rojo).

2. **Mapa de subsidencia**: capa de velocidad de subsidencia (mm/año) del InSAR,
   actualizada con cada nueva adquisición.

3. **Panel de reporte individual**: al hacer clic en un marcador, muestra:
   - Foto original y ortofoto lado a lado
   - Vector `v_grieta` con barras de progreso
   - Gráfica temporal de evolución de la grieta
   - Serie temporal de subsidencia del punto

4. **Lista de prioridades**: tabla ordenada por `risk_index DESC` con filtros
   por colonia, fecha, severidad y estado de atención.

5. **Alertas automáticas**: si `s̈ > umbral_aceleración` en una zona nueva,
   generar alerta aunque no haya reporte ciudadano aún (predicción preventiva).

---

## Dependencias globales del proyecto

```
# Dataset y preprocesamiento
requests Pillow pandas tqdm numpy

# Entrenamiento
torch torchvision transformers scikit-learn matplotlib

# Inferencia y backend
fastapi uvicorn[standard] onnxruntime opencv-python-headless
sqlalchemy asyncpg geoalchemy2 psycopg2-binary celery redis

# InSAR y geoespacial
rasterio geopandas shapely scipy

# Segmentación de grietas
# DeepCrack: instalar desde https://github.com/yhlleo/DeepCrack
```

---

## Orden de implementación recomendado

```
Fase 0  ✅  →  Fase 1  →  Fase 2  →  Fase 3
                                         ↓
Fase 4  (paralelo con Fase 2–3)    Fase 5 (backend shell)
  ↓                                      ↓
  └──────────────────────┬───────────────┘
                         ↓
                      Fase 6  →  Fase 7
                                    ↓
                      Fase 8  (cuando haya datos InSAR disponibles)
                                    ↓
                      Fase 9  (cuando Fase 5 + 7 + 8 estén listos)
```

Fases 4 y 2–3 pueden desarrollarse en paralelo por equipos distintos porque
sus interfaces están bien definidas (imagen JPEG + JSON sidecar → API endpoint).

---

## Información para retomar en un chat nuevo

Para retomar cualquier fase en un chat nuevo, compartir este archivo TODO.md
es suficiente. No se necesita contexto adicional de conversaciones anteriores.

Cada fase incluye: objetivo, entradas, salidas, código de referencia con
firmas de funciones, hiperparámetros fijados, criterios de aceptación y
dependencias. La sección "Decisiones de diseño fijadas" evita reabrir
discusiones ya resueltas.

El archivo `contexto.md` en el mismo repositorio contiene la descripción
general del problema y la arquitectura de alto nivel del sistema completo.
