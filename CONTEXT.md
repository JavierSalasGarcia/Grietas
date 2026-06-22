# CONTEXT.md — Grietas Toluca · Referencia rápida de contexto

> **Instrucción para el modelo**: Lee este archivo al inicio de cada chat de
> implementación. Contiene todos los nombres canónicos de archivos, carpetas,
> variables y constantes usados en el proyecto. No inventar variantes distintas.

---

## 1. Descripción del proyecto (3 líneas)

Sistema ciudadano de reporte de grietas en pavimentos de Toluca, México.
Las fotos oblicuas (45°) se ortorrectifican, segmentan y clasifican por severidad.
Un índice de riesgo R ∈ [0,10] combina métricas de grieta con subsidencia InSAR de
Sentinel-1 (2014–2026) y se visualiza en un panel municipal.

---

## 2. Estado de fases

| Fase | Nombre | Estado | Chat |
|------|--------|--------|------|
| 0 | Descarga de datasets | ✅ Completo | — |
| 1 | Preprocesamiento | ⬜ Pendiente | chat nuevo |
| 2 | Entrenamiento teacher (DINOv2) | ⬜ Pendiente | chat nuevo (depende Fase 1) |
| 3 | Entrenamiento student (MobileNetV3) | ⬜ Pendiente | chat nuevo (depende Fase 2) |
| 4 | PWA ciudadana | ⬜ Pendiente | chat nuevo (independiente) |
| 5 | Backend FastAPI | ✅ Completo 2026-06-22 | rama `claude/grietas-toluca-phase-5-39vnvl` |
| 6 | Ortorrectificación | ⬜ Pendiente | chat nuevo (depende Fase 5) |
| 7 | Segmentación y análisis | ⬜ Pendiente | chat nuevo (depende Fase 6) |
| 8 | InSAR + índice de riesgo | ⬜ Pendiente | chat nuevo (depende Fase 7) |
| 9 | Dashboard municipal | ⬜ Pendiente | chat nuevo (depende Fase 8) |

**Actualizar esta tabla al completar cada fase** (cambiar ⬜ → ✅ y añadir fecha).

---

## 3. Estructura de carpetas canónica

```
Grietas/                            ← raíz del repositorio
│
├── CONTEXT.md                      ← este archivo (contexto para cada chat)
├── TODO.md                         ← notas técnicas de diseño
├── contexto.md                     ← descripción narrativa del proyecto
├── fase0.md … fase9.md             ← especificaciones completas por fase
│
├── download_crack_datasets.py      ← Fase 0 ✅
│
├── preprocessing/                  ← Fase 1
│   └── preprocess_dataset.py
│
├── training/                       ← Fases 2 y 3
│   ├── train_teacher.py
│   └── train_student.py
│
├── pwa/                            ← Fase 4 (app ciudadana)
│   ├── index.html
│   ├── manifest.json
│   ├── sw.js
│   └── src/
│       ├── orientation.js
│       ├── geolocation.js
│       ├── ua-camera-db.js
│       ├── citizen-store.js
│       ├── camera.js
│       └── upload.js
│
├── api/                            ← Fases 5–8 (backend FastAPI)
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── classifier.py
│   ├── tasks.py
│   ├── orthorectify.py             ← Fase 6
│   ├── crack_analysis.py           ← Fase 7
│   ├── insar.py                    ← Fase 8
│   ├── risk.py                     ← Fase 8
│   └── tiles.py                    ← Fase 9 (tile server InSAR)
│
├── dashboard/                      ← Fase 9 (panel municipal)
│   ├── index.html
│   ├── manifest.json
│   ├── sw.js
│   └── src/
│       ├── api-client.js
│       ├── map.js
│       ├── heatmap.js
│       ├── insar-layer.js
│       ├── report-panel.js
│       ├── priority-list.js
│       ├── alerts.js
│       └── charts.js
│
├── tests/                          ← tests unitarios (pytest)
│   ├── test_preprocess.py
│   ├── test_orthorectify.py
│   └── test_crack_analysis.py
│
├── models/                         ← artefactos entrenados (gitignored)
│   ├── teacher_dinov2_best.pth
│   ├── teacher_dinov2_config.json
│   ├── student_mobilenetv3_best.pth
│   ├── student_mobilenetv3.onnx     ← ≤ 15 MB, opset 17
│   └── student_config.json
│
├── dataset_processed/              ← salida de Fase 1 (gitignored)
│   ├── train/
│   │   ├── Aprobado/
│   │   └── No_Aprobado/
│   ├── val/
│   │   ├── Aprobado/
│   │   └── No_Aprobado/
│   └── test/
│       ├── Aprobado/
│       └── No_Aprobado/
│
├── data/                           ← datos InSAR (gitignored)
│   └── insar_velocity.tif
│
└── pavement_crack_datasets/        ← descarga Fase 0 (gitignored)
    ├── roboflow/
    ├── mendeley/
    ├── github/
    ├── figshare/
    ├── coco_val2017/
    ├── registry.csv
    └── registry.json
```

---

## 4. Nombres canónicos de variables y constantes

### Clasificación (Fases 2–3–5)

| Nombre | Valor | Dónde |
|--------|-------|-------|
| `CAT_APROBADO` | `"Aprobado"` | preprocessing, training |
| `CAT_NO_APROBADO` | `"No_Aprobado"` | preprocessing, training |
| `IMG_SIZE` | `(224, 224)` | preprocessing, dataloaders |
| `FROZEN_BLOCKS` | `10` | train_teacher.py |
| `LR_BACKBONE` | `1e-5` | train_teacher.py |
| `LR_HEAD` | `1e-3` | train_teacher.py |
| `TEMPERATURE` | `3` | train_student.py (destilación) |
| `ALPHA` | `0.5` | train_student.py (peso CE vs KL) |
| `MAX_IMBALANCE_RATIO` | `3.0` | preprocess_dataset.py |
| `RANDOM_SEED` | `42` | todo el proyecto |
| `SPLIT_TRAIN` | `0.70` | preprocess_dataset.py |
| `SPLIT_VAL` | `0.15` | preprocess_dataset.py |
| `SPLIT_TEST` | `0.15` | preprocess_dataset.py |

### Ortorrectificación (Fase 6)

| Nombre | Valor | Descripción |
|--------|-------|-------------|
| `PITCH_MIN_DEG` | `30.0` | límite inferior para quality="good" |
| `PITCH_MAX_DEG` | `60.0` | límite superior para quality="good" |
| `MAX_OUTPUT_PX` | `2048` | tamaño máximo ortofoto de salida |
| `DEFAULT_FOV_DEG` | `70.0` | FoV por defecto si no viene en metadata |

### Análisis de grieta (Fase 7)

| Nombre | Valor | Descripción |
|--------|-------|-------------|
| `SKELETON_THRESHOLD` | `0.5` | umbral sigmoid para máscara DeepCrack |
| `SEVERITY_LEVE_W` | `0.3` mm | ancho máximo para leve |
| `SEVERITY_MOD_W` | `3.0` mm | ancho máximo para moderado |
| `SEVERITY_LEVE_A` | `0.5` % | área máxima para leve |
| `SEVERITY_MOD_A` | `5.0` % | área máxima para moderado |

### Riesgo e InSAR (Fase 8)

| Nombre | Valor | Descripción |
|--------|-------|-------------|
| `LOS_TO_VERTICAL` | `1 / cos(38°) ≈ 1.269` | conversión LOS → vertical |
| `INSAR_RADIUS_M` | `30` | radio búsqueda InSAR en metros |
| `RISK_ALPHA` | `0.40` | peso severidad en R(x,y) |
| `RISK_BETA` | `0.30` | peso velocidad de subsidencia |
| `RISK_GAMMA` | `0.20` | peso aceleración de subsidencia |
| `RISK_DELTA` | `0.10` | peso densidad de reportes cercanos |
| `ALERT_RISK_THRESHOLD` | `8.0` | risk_index mínimo para alerta |
| `ALERT_ACCEL_THRESHOLD` | `15.0` | mm/año² para alerta de aceleración |
| `ALERT_CLUSTER_RADIUS_M` | `100` | radio de clúster de alertas |
| `ALERT_CLUSTER_COUNT` | `3` | mínimo de reportes severos en clúster |

---

## 5. Schema del JSON sidecar (PWA → backend)

```json
{
  "schema_version": "1.0",
  "captured_at": "ISO-8601",
  "citizen": {
    "height_cm": 170,
    "device_id": "uuid-v4",
    "user_agent": "Mozilla/5.0 …"
  },
  "camera": {
    "orientation": {
      "alpha": 90.0,
      "beta": 45.0,
      "gamma": 0.0,
      "absolute": true
    },
    "resolution": { "width": 4000, "height": 3000 },
    "fov_estimate_deg": 70.0,
    "intrinsics_source": "ua_database"
  },
  "gps": {
    "lat": 19.282,
    "lng": -99.654,
    "alt_m": 2660.0,
    "accuracy_m": 5.0
  },
  "previous_report_id": null
}
```

`intrinsics_source` acepta: `"webxr"` | `"onboarding_calibration"` | `"ua_database"` | `"prior"`

---

## 6. Vector de severidad de grieta `v_grieta`

```python
v_grieta = [
    w_mean_mm,        # ancho medio en mm
    length_mm,        # longitud del esqueleto en mm
    area_mm2,         # área de la máscara en mm²
    fractal_dim,      # dimensión fractal D_f ∈ [1.0, 2.0]
    orientation_deg,  # ángulo dominante en grados (0–180)
    area_ratio,       # área_grieta / área_imagen (fracción)
    branch_points,    # número de nodos de ramificación
    severity_class,   # "leve" | "moderado" | "severo"
    severity_score,   # escalar ∈ [0.0, 10.0]
]
```

---

## 7. Índice de riesgo R

```
R(x,y,t) = 0.40 · severity_score_norm
          + 0.30 · |velocity_norm|
          + 0.20 · max(acceleration_norm, 0)
          + 0.10 · density_norm
∈ [0, 10]
```

Colores de riesgo en dashboard:
- `[8, 10]` → `#d32f2f` rojo — Intervención **inmediata**
- `[5,  8)` → `#f57c00` naranja — prioridad **alta**
- `[3,  5)` → `#fbc02d` amarillo — prioridad **media**
- `[0,  3)` → `#388e3c` verde — prioridad **baja**

---

## 8. Modelos de IA

| Rol | Modelo | Tamaño | Exportación |
|-----|--------|--------|-------------|
| Teacher | `facebook/dinov2-base` (ViT-B/14) | 86 M params | `.pth` |
| Student | `torchvision.models.mobilenet_v3_small` | 2.5 M params | `.onnx` ≤ 15 MB |
| Segmentación | DeepCrack (Liu et al. 2019) | — | `.pth` |
| Fallback seg. | CLAHE + adaptiveThreshold + morphology | — | OpenCV |

Inferencia en producción: ONNX Runtime, `CPUExecutionProvider`, latencia ≤ 200 ms.

---

## 9. Infraestructura backend

| Componente | Tecnología | Notas |
|------------|-----------|-------|
| API | FastAPI (async) | puerto 8000 |
| Cola de tareas | Celery + Redis | broker redis://localhost:6379/0 |
| Base de datos | PostgreSQL 15 + PostGIS 3 | puerto 5432 |
| ORM | SQLAlchemy 2 async (asyncpg) | |
| Tile server | rio-tiler + rasterio | capa InSAR |
| Assets estáticos | `StaticFiles` de FastAPI | `/dashboard/` |

Variables de entorno requeridas:
```
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/grietas
REDIS_URL=redis://localhost:6379/0
MODELS_DIR=./models
IMAGES_DIR=./images_storage
ROBOFLOW_API_KEY=...     # solo para Fase 0
GITHUB_TOKEN=...          # opcional para Fase 0
```

---

## 10. Ramas git y convenciones

### Estrategia de ramas

Cada fase se desarrolla en su propia rama y se mergea a `main` al pasar tests.

```
main  (siempre ejecutable)
├── fase/1-preprocessing        base: main
├── fase/2-teacher              base: main (mergea tras fase/1)
├── fase/3-student              base: main (mergea tras fase/2)
├── fase/4-pwa                  base: main (independiente)
├── fase/5-backend              base: main (independiente)
├── fase/6-orthorectify         base: main (mergea tras fase/5)
├── fase/7-crack-analysis       base: main (mergea tras fase/6)
├── fase/8-insar-risk           base: main (mergea tras fase/7)
└── fase/9-dashboard            base: main (mergea tras fase/8)
```

Las fases **1–3** y **4–5** pueden desarrollarse en paralelo entre sí;
las fases **6–9** son secuenciales y requieren que la anterior esté en `main`.

### Convenciones de commit

- `feat: fase N — descripción` al terminar la implementación principal
- `test: fase N — descripción` al añadir/ajustar tests
- `fix: fase N — descripción` para correcciones post-revisión

### Flujo por fase

```bash
git checkout main && git pull origin main
git checkout -b fase/N-nombre
# ... desarrollo ...
pytest tests/ -v                          # todos los tests en verde
git add <archivos específicos>
git commit -m "feat: fase N — descripción"
git push -u origin fase/N-nombre
# crear PR → revisar → merge a main
```

- **Tests**: `pytest tests/ -v` antes de cada commit
- **Archivos gitignored**: `pavement_crack_datasets/`, `dataset_processed/`, `models/`, `data/`, `__pycache__/`, `.env`

---

## 11. Cómo iniciar el chat de cada fase

Pegar al inicio del nuevo chat:

```
Implementa la Fase N del proyecto Grietas Toluca.
Lee CONTEXT.md y faseN.md para el contexto completo.
Crea la rama fase/N-<nombre> con base en main.
Repositorio: javiersalasgarcia/grietas
```

El modelo leerá ambos archivos y tendrá todo lo necesario para proceder
sin preguntas adicionales de contexto.
