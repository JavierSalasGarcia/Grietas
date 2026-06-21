# Contexto del Proyecto — Clasificador de Grietas en Calles de Toluca

## Descripción del Problema

Se está desarrollando un sitio web que permite a ciudadanos reportar grietas y fisuras
en las calles de Toluca, subiendo fotografías. Para filtrar reportes inválidos (fotos
fuera de tema, bromas, errores del usuario), se necesita un **modelo de clasificación
binaria** que decida automáticamente si una imagen contiene una grieta real o no.

### Clases del modelo

| Clase | Descripción |
|---|---|
| **Aprobado** | La imagen muestra una grieta o fisura real en pavimento |
| **No Aprobado** | Foto aleatoria, objeto cotidiano, error del usuario, broma |

---

## Estado Actual del Proyecto

### Lo que ya existe

- **Rama de desarrollo:** `claude/pavement-crack-downloader-ywz2ai`
- **Repositorio remoto:** `javiersalasgarcia/grietas`
- **Archivo principal:** `download_crack_datasets.py`
- **Archivo de ignorados:** `.gitignore` (excluye `pavement_crack_datasets/`)

### Lo que hace `download_crack_datasets.py`

Script Python modularizado que construye el dataset automáticamente descargando
desde 5 fuentes públicas:

```
[Aprobado — grietas reales]
1. Roboflow Universe   → pavement-crack-detection-r4n7n  (requiere API key gratis)
2. Mendeley Data       → dataset 88kdyyc73h v1            (sin autenticación)
3. GitHub              → juhuyan/CrackDataset_DL_HY       (sin autenticación)
4. Figshare            → article 25103138                  (sin autenticación)

[No Aprobado — objetos cotidianos]
5. COCO val2017        → ~5 000 imágenes aleatorias        (sin autenticación)
```

### Estructura de salida generada por el script

```
pavement_crack_datasets/
├── 01_roboflow/
│   ├── roboflow_yolov8_v1.zip
│   └── extracted/
│       ├── train/images/   ← imágenes de grietas
│       ├── train/labels/   ← anotaciones YOLOv8
│       ├── valid/
│       └── test/
├── 02_mendeley/            ← archivos del dataset Mendeley
├── 03_github/              ← estructura del repo GitHub
├── 04_figshare/            ← archivos de Figshare (ZIP extraídos)
├── 05_coco_control/
│   ├── images/             ← fotos de objetos cotidianos (No Aprobado)
│   └── annotations/        ← un JSON por imagen con sus anotaciones COCO
├── download_registry.csv   ← índice consolidado (abre con Excel o pandas)
├── download_registry.json  ← mismo índice en JSON
└── download.log            ← log detallado de todas las operaciones
```

### Esquema del registro consolidado (`download_registry.csv`)

| Campo | Descripción |
|---|---|
| `id` | UUID único por fila |
| `source` | roboflow / mendeley / github / figshare / coco |
| `dataset_name` | nombre específico del dataset |
| `filename` | nombre del archivo local |
| `local_path` | ruta absoluta en disco |
| `source_url` | URL de origen de la descarga |
| `file_type` | image / annotation / archive / other |
| `categoria_proyecto` | **Aprobado** o **No Aprobado** |
| `label` | etiqueta semántica o nombre de carpeta |
| `annotation_file` | ruta del archivo de anotación asociado |
| `md5` | checksum para verificar integridad |
| `size_bytes` | tamaño del archivo |
| `downloaded_at` | timestamp UTC de la descarga |

---

## Cómo Ejecutar el Script

### Instalación de dependencias

```bash
pip install requests
```

Solo se necesita `requests`. No se requiere `pycocotools` ni ninguna otra librería.

### Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `ROBOFLOW_API_KEY` | **Sí** (para fuente 1) | Clave de API de Roboflow. Gratis en https://app.roboflow.com/ |
| `GITHUB_TOKEN` | No (recomendada) | Token de GitHub. Sube el límite de 60 a 5 000 peticiones/hora |

### Ejecución

```bash
export ROBOFLOW_API_KEY=rf_xxxxxxxxxxxx
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx   # opcional

python download_crack_datasets.py
```

### Tiempo estimado de descarga

| Fuente | Tamaño aprox. | Tiempo aprox. |
|---|---|---|
| Roboflow | 50–500 MB | 2–10 min |
| Mendeley | variable | variable |
| GitHub | variable | variable |
| Figshare | variable | variable |
| COCO val2017 (imágenes + anots) | ~1.2 GB | 20–60 min |

---

## Próximos Pasos del Proyecto

El pipeline completo desde los datos crudos hasta el modelo en producción sigue
estos pasos. Los marcados con ✅ ya están implementados.

### Fase 1 — Recolección de datos ✅

- [x] Script de descarga `download_crack_datasets.py`
- [x] Manejo de errores y reintentos automáticos
- [x] Nombres únicos para evitar sobreescritura
- [x] Registro consolidado CSV + JSON
- [x] Categorías "Aprobado" / "No Aprobado"

### Fase 2 — Preprocesamiento (pendiente)

- [ ] Leer el `download_registry.csv` con pandas y filtrar solo `file_type == "image"`
- [ ] Verificar integridad de imágenes (detectar archivos corruptos con `PIL.Image.verify()`)
- [ ] Redimensionar todas las imágenes a un tamaño uniforme (p. ej. 224×224 o 256×256)
- [ ] Dividir en conjuntos train / val / test (p. ej. 70/15/15)
- [ ] Balancear las clases (actualmente habrá más imágenes "Aprobado" que "No Aprobado"
  o viceversa; se puede submuestrear o usar pesos de clase)
- [ ] Guardar los splits en archivos CSV o como estructura de carpetas estilo ImageFolder

### Fase 3 — Entrenamiento del modelo (pendiente)

Opciones recomendadas (de más simple a más potente):

#### Opción A — Transfer Learning con PyTorch/torchvision (recomendado para empezar)
```python
# Modelo base: ResNet-18 o EfficientNet-B0 preentrenados en ImageNet
# Fine-tuning de la última capa para clasificación binaria
# Librerías: torch, torchvision, Pillow
```

#### Opción B — Transfer Learning con Keras/TensorFlow
```python
# Modelo base: MobileNetV2 o EfficientNetB0
# Ligero y apto para producción web
# Librerías: tensorflow, keras
```

#### Opción C — Usar el modelo YOLOv8 de Roboflow directamente
```python
# Si el objetivo es detección (bounding box) además de clasificación
# pip install ultralytics
# model = YOLO("yolov8n-cls.pt")  # versión de clasificación
```

**Métricas clave a monitorear:**
- Accuracy en val y test
- Precision y Recall para la clase "Aprobado" (evitar falsos negativos)
- F1-score
- Matriz de confusión

### Fase 4 — Evaluación y ajuste (pendiente)

- [ ] Analizar errores: ¿qué tipo de imágenes confunde el modelo?
- [ ] Data augmentation si el accuracy en val es bajo (rotaciones, brillo, zoom)
- [ ] Ajustar threshold de decisión si se necesita más recall que precision
- [ ] Exportar modelo en formato ONNX para portabilidad

### Fase 5 — Despliegue en el sitio web (pendiente)

- [ ] Envolver el modelo en una API REST (Flask o FastAPI)
- [ ] Endpoint `POST /clasificar` que recibe la imagen y devuelve `{"resultado": "Aprobado", "confianza": 0.94}`
- [ ] Integrar con el formulario de reporte ciudadano
- [ ] Guardar en base de datos: imagen, resultado, confianza, timestamp

---

## Decisiones de Diseño Tomadas

| Decisión | Razón |
|---|---|
| Solo `requests`, sin `pycocotools` | Elimina dependencia nativa; COCO se parsea directamente con `json` |
| COCO val2017 (no train2017) | Tiene exactamente 5 000 imágenes, tamaño manejable, representativas |
| Semilla aleatoria fija para COCO | Reproducibilidad: siempre se seleccionan las mismas imágenes |
| JSON por imagen en COCO | Facilita el acceso por imagen sin cargar el JSON entero de 250 MB |
| Formato YOLOv8 en Roboflow | Compatible con el ecosistema Ultralytics, ampliamente usado |
| Prefijo `coco_` en nombres de archivo | Evita colisiones de nombres entre fuentes |

---

## Comandos Git Útiles

```bash
# Ver el estado del repositorio
git status

# Cambiar a la rama de desarrollo
git checkout claude/pavement-crack-downloader-ywz2ai

# Ver el historial de commits
git log --oneline

# Subir cambios
git add download_crack_datasets.py
git commit -m "descripción del cambio"
git push origin claude/pavement-crack-downloader-ywz2ai
```

---

## Preguntas Frecuentes

**¿Por qué COCO y no otro dataset de "No Aprobado"?**
COCO es el estándar de facto en visión por computadora, está bien documentado,
es gratuito y cubre una diversidad enorme de escenas cotidianas sin ninguna
imagen de grietas o pavimento dañado.

**¿Cuántas imágenes necesito para entrenar?**
Con transfer learning, 500–1 000 imágenes por clase suelen ser suficientes para
obtener buenos resultados. Las 4 fuentes de grietas juntas deberían superar esa
cifra; COCO aporta 5 000 negativos.

**¿Qué pasa si Mendeley o Figshare cambian su API?**
El script registra el error en el log y continúa con las demás fuentes.
Los mensajes de error incluyen la URL de descarga manual como alternativa.

**¿El script es reanudable?**
Parcialmente. Usa `unique_path()` para no sobreescribir archivos ya descargados,
pero no tiene un estado de progreso persistente. Si se interrumpe, los archivos ya
descargados se conservan pero el proceso arranca de nuevo desde la primera fuente.
Para datasets muy grandes conviene implementar un checkpoint (próxima mejora sugerida).
