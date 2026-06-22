# Fase 0 — Construcción del Dataset `[COMPLETADA ✅]`

## Contexto del proyecto

Sistema de reporte ciudadano de grietas en calles de Toluca. Un modelo de IA
binario clasifica cada fotografía enviada por ciudadanos como **Aprobado**
(grieta real en pavimento) o **No Aprobado** (foto inválida, broma, error).
Esta fase construye el dataset de entrenamiento descargando imágenes de fuentes
públicas.

---

## Objetivo de la fase

Descargar automáticamente imágenes de grietas y de objetos cotidianos desde
cinco fuentes públicas, organizarlas en carpetas separadas, y producir un
registro consolidado en CSV y JSON que sirva como índice para las fases siguientes.

---

## Resultado: qué se construyó

### Archivo principal
```
download_crack_datasets.py    ← script Python autocontenido
```

### Estructura de salida generada
```
pavement_crack_datasets/
  01_roboflow/
    roboflow_yolov8_v1.zip         ← archivo original descargado
    extracted/
      train/
        images/                    ← imágenes de grietas (Aprobado)
        labels/                    ← anotaciones YOLOv8 (.txt, un archivo por imagen)
      valid/
        images/
        labels/
      test/
        images/
        labels/
  02_mendeley/                     ← archivos del dataset 88kdyyc73h
  03_github/                       ← estructura original del repo CrackDataset_DL_HY
  04_figshare/                     ← archivos del artículo 25103138 (ZIPs extraídos)
  05_coco_control/
    images/                        ← 5 000 imágenes cotidianas (No Aprobado)
    annotations/                   ← un JSON por imagen (categorías COCO, bboxes)
      instances_val2017.json       ← anotaciones maestras COCO val2017
      coco_000000001000.json       ← anotación individual por imagen
      ...
  download_registry.csv            ← índice consolidado
  download_registry.json           ← mismo índice en JSON
  download.log                     ← log detallado de todas las operaciones
```

### Esquema del registro consolidado

Cada fila del CSV/JSON representa un archivo descargado:

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | UUID | Identificador único de la fila |
| `source` | str | `roboflow` / `mendeley` / `github` / `figshare` / `coco` |
| `dataset_name` | str | Nombre específico del dataset de origen |
| `filename` | str | Nombre del archivo en disco |
| `local_path` | str | Ruta absoluta del archivo en disco |
| `source_url` | str | URL desde donde se descargó |
| `file_type` | str | `image` / `annotation` / `archive` / `other` |
| `categoria_proyecto` | str | **`Aprobado`** o **`No Aprobado`** |
| `label` | str | Nombre de carpeta o categorías COCO del archivo |
| `annotation_file` | str | Ruta del JSON de anotación asociado (si existe) |
| `md5` | str | Checksum MD5 para verificar integridad |
| `size_bytes` | int | Tamaño del archivo en bytes |
| `downloaded_at` | str | ISO 8601 UTC del momento de descarga |

---

## Fuentes descargadas

| # | Fuente | URL | Categoría | Auth |
|---|---|---|---|---|
| 1 | Roboflow | `universe.roboflow.com/mcmaster-vkq5k/pavement-crack-detection-r4n7n` | Aprobado | `ROBOFLOW_API_KEY` (gratis) |
| 2 | Mendeley Data | `data.mendeley.com/datasets/88kdyyc73h/1` | Aprobado | Ninguna |
| 3 | GitHub | `github.com/juhuyan/CrackDataset_DL_HY` | Aprobado | `GITHUB_TOKEN` (opcional) |
| 4 | Figshare | `figshare.com/articles/dataset/.../25103138` | Aprobado | Ninguna |
| 5 | COCO val2017 | `images.cocodataset.org/val2017/` | No Aprobado | Ninguna |

---

## Cómo ejecutar

```bash
# 1. Instalar dependencia única
pip install requests

# 2. Configurar credenciales
export ROBOFLOW_API_KEY=rf_xxxxxxxxxxxx   # obligatorio para fuente 1
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx      # opcional, sube límite a 5 000 req/h

# 3. Ejecutar desde la raíz del repositorio
python download_crack_datasets.py
```

Tiempo estimado total: **30–90 minutos** según velocidad de conexión.
El descargador de COCO (~1.2 GB de imágenes) domina el tiempo total.

---

## Cómo verificar que la fase está completa

```python
import pandas as pd

df = pd.read_csv("pavement_crack_datasets/download_registry.csv")

# Verificaciones básicas
assert len(df) > 0,                  "Registro vacío"
assert "categoria_proyecto" in df.columns, "Falta columna de categoría"

imagenes  = df[df["file_type"] == "image"]
aprobado  = imagenes[imagenes["categoria_proyecto"] == "Aprobado"]
no_aprobado = imagenes[imagenes["categoria_proyecto"] == "No Aprobado"]

print(f"Total imágenes: {len(imagenes)}")
print(f"  Aprobado:     {len(aprobado)}")
print(f"  No Aprobado:  {len(no_aprobado)}")

# El dataset mínimo viable requiere al menos:
assert len(aprobado)   >= 500,  "Pocas imágenes de grietas"
assert len(no_aprobado) >= 500, "Pocas imágenes de control"
```

---

## Siguiente paso

**Fase 1** — Preprocesamiento y validación (`fase1.md`).
Lee `download_registry.csv`, verifica integridad de imágenes, redimensiona
a 224×224, balancea clases y divide en train/val/test.

---

## Notas para futuras ejecuciones

- El script es **idempotente para COCO**: si las imágenes ya existen en disco,
  `unique_path()` les añade sufijo `_1`, `_2`, etc. en lugar de sobreescribir.
  Para una re-descarga limpia, eliminar `pavement_crack_datasets/` completo.
- La carpeta `pavement_crack_datasets/` está en `.gitignore` y **no se versiona**.
  Puede pesar varios GB; almacenarla en un volumen externo o bucket S3.
- Si Mendeley o Figshare cambian su API, el script registra el error y continúa
  con las demás fuentes. Revisar `download.log` para diagnóstico.
