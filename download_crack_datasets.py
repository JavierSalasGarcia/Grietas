#!/usr/bin/env python3
"""
Dataset Builder — Clasificador de Grietas en Calles de Toluca
=============================================================
Construye automáticamente el dataset para un modelo de clasificación binaria
que decide si una imagen enviada por un ciudadano contiene una grieta/fisura
real en el pavimento.

Categorías del proyecto:
  • "Aprobado"     → imagen con grieta o fisura real en pavimento
  • "No Aprobado"  → foto aleatoria, objeto cotidiano, error o broma del usuario

Fuentes descargadas:
  [Aprobado]
  1. Roboflow Universe  — pavement-crack-detection-r4n7n
  2. Mendeley Data      — dataset 88kdyyc73h (v1)
  3. GitHub             — juhuyan/CrackDataset_DL_HY
  4. Figshare           — article 25103138

  [No Aprobado]
  5. COCO val2017       — ~5 000 imágenes de objetos y escenas cotidianas

Salida:
  pavement_crack_datasets/
  ├── 01_roboflow/          # grietas — Aprobado
  ├── 02_mendeley/          # grietas — Aprobado
  ├── 03_github/            # grietas — Aprobado
  ├── 04_figshare/          # grietas — Aprobado
  ├── 05_coco_control/      # objetos cotidianos — No Aprobado
  ├── download_registry.csv # índice consolidado
  ├── download_registry.json
  └── download.log

Instalación de dependencias:
  pip install requests

Variables de entorno (opcionales/requeridas):
  ROBOFLOW_API_KEY  → API key gratuita de Roboflow (obligatoria para fuente 1)
  GITHUB_TOKEN      → Personal Access Token de GitHub (sube límite a 5 000 req/h)

Uso:
  export ROBOFLOW_API_KEY=<tu_clave>
  python download_crack_datasets.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# Módulos de la biblioteca estándar
# ──────────────────────────────────────────────────────────────────────────────
import csv
import hashlib
import json
import logging
import os
import random
import sys
import tarfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias externas (solo 'requests')
# ──────────────────────────────────────────────────────────────────────────────
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print(
        "ERROR: la librería 'requests' no está instalada.\n"
        "Instálala con:  pip install requests"
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

# Directorio raíz de todas las descargas
BASE_DIR = Path("pavement_crack_datasets")

# Carpeta por fuente  (prefijo numérico = orden de ejecución)
SOURCE_DIRS: Dict[str, Path] = {
    "roboflow":     BASE_DIR / "01_roboflow",
    "mendeley":     BASE_DIR / "02_mendeley",
    "github":       BASE_DIR / "03_github",
    "figshare":     BASE_DIR / "04_figshare",
    "coco_control": BASE_DIR / "05_coco_control",
}

# Categorías del proyecto
CAT_APROBADO     = "Aprobado"
CAT_NO_APROBADO  = "No Aprobado"

# ── Credenciales (leídas de variables de entorno) ─────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN", "")

# ── Parámetros Roboflow ───────────────────────────────────────────────────────
ROBOFLOW_WORKSPACE = "mcmaster-vkq5k"
ROBOFLOW_PROJECT   = "pavement-crack-detection-r4n7n"
ROBOFLOW_VERSION   = 1
# Formatos disponibles: yolov8, yolov5, coco, pascal-voc, tensorflow, retinanet
ROBOFLOW_FORMAT    = "yolov8"

# ── Parámetros Mendeley Data ──────────────────────────────────────────────────
MENDELEY_DATASET_ID = "88kdyyc73h"
MENDELEY_VERSION    = 1

# ── Parámetros GitHub ─────────────────────────────────────────────────────────
GITHUB_REPO = "juhuyan/CrackDataset_DL_HY"

# ── Parámetros Figshare ───────────────────────────────────────────────────────
FIGSHARE_ARTICLE_ID = 25103138

# ── Parámetros COCO ───────────────────────────────────────────────────────────
# Usamos el conjunto de validación 2017 (exactamente 5 000 imágenes, ~1 GB)
COCO_MAX_IMAGES      = 5_000          # 0 = descargar todas
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_BASE_URL  = "http://images.cocodataset.org/val2017/"
COCO_RANDOM_SEED     = 42             # semilla para reproducibilidad

# ── Parámetros HTTP ───────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 120          # segundos por petición
RETRY_ATTEMPTS  = 4            # reintentos automáticos ante errores 5xx / 429
RETRY_BACKOFF   = 2.0          # segundos de espera base (crece exponencialmente)
CHUNK_SIZE      = 64 * 1024    # 64 KB por chunk al escribir en disco

# Límite de tamaño de archivo (bytes). 0 = sin límite.
MAX_FILE_BYTES: int = 0

# ── Extensiones reconocidas ───────────────────────────────────────────────────
IMAGE_EXTS: Set[str]      = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_EXTS: Set[str] = {".xml", ".json", ".txt", ".csv", ".yaml", ".yml"}
ARCHIVE_EXTS: Set[str]    = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CONFIGURACIÓN DE LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """
    Configura dos handlers:
      • Archivo  download.log  → nivel DEBUG (registro detallado)
      • Consola  stdout        → nivel INFO  (resumen visible al usuario)
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("grietas_dl")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de archivo (registro completo)
    file_handler = logging.FileHandler(BASE_DIR / "download.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Handler de consola (solo mensajes importantes)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — SESIÓN HTTP CON REINTENTOS AUTOMÁTICOS
# ══════════════════════════════════════════════════════════════════════════════

def build_session(extra_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    """
    Crea una sesión HTTP con:
      • Reintentos automáticos con backoff exponencial ante errores 5xx y 429
      • User-Agent personalizado
      • Headers adicionales opcionales (p. ej. Authorization de GitHub)
    """
    session = requests.Session()

    retry_policy = Retry(
        total=RETRY_ATTEMPTS,
        backoff_factor=RETRY_BACKOFF,
        # Reintentar solo ante estos códigos HTTP
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({"User-Agent": "GrietasTolucaDownloader/2.0"})
    if extra_headers:
        session.headers.update(extra_headers)

    return session


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════

def unique_path(directory: Path, filename: str) -> Path:
    """
    Devuelve una ruta libre de colisiones.
    Si `directory/filename` ya existe, añade _1, _2, … al stem hasta encontrar
    un nombre disponible. Garantiza que nunca se sobreescriba un archivo.
    """
    stem   = Path(filename).stem
    suffix = Path(filename).suffix
    dest   = directory / filename
    n = 1
    while dest.exists():
        dest = directory / f"{stem}_{n}{suffix}"
        n += 1
    return dest


def md5_checksum(path: Path) -> str:
    """Calcula el MD5 de un archivo para verificar integridad."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    """Convierte bytes a unidad legible (KB, MB, GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_file(
    url: str,
    dest: Path,
    session: requests.Session,
    logger: logging.Logger,
) -> bool:
    """
    Descarga `url` a `dest` en modo streaming (no carga el archivo completo
    en memoria). Captura y registra cada tipo de error por separado.
    Elimina el archivo parcial si la descarga falla.

    Retorna True si la descarga fue exitosa, False en caso contrario.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()

            # Respetar el límite de tamaño configurado
            content_length = int(resp.headers.get("Content-Length", 0))
            if MAX_FILE_BYTES and content_length > MAX_FILE_BYTES:
                logger.warning(
                    "Omitido — tamaño %s supera el límite %s: %s",
                    human_size(content_length), human_size(MAX_FILE_BYTES), url,
                )
                return False

            downloaded = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

        logger.debug("✓ %s  (%s) → %s", url, human_size(downloaded), dest.name)
        return True

    except requests.HTTPError as exc:
        logger.error("HTTP %s al descargar: %s", exc.response.status_code, url)
    except requests.ConnectionError:
        logger.error("Error de conexión: %s", url)
    except requests.Timeout:
        logger.error("Timeout: %s", url)
    except OSError as exc:
        logger.error("Error de escritura en %s: %s", dest, exc)

    # Limpiar archivo parcial para no dejar datos corruptos
    if dest.exists():
        dest.unlink(missing_ok=True)
    return False


def extract_archive(archive: Path, dest_dir: Path, logger: logging.Logger) -> bool:
    """
    Extrae un archivo ZIP o TAR (incluyendo .gz, .bz2, .xz) en `dest_dir`.
    Retorna True si la extracción fue exitosa.
    """
    suffix_chain = "".join(archive.suffixes).lower()
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest_dir)
            logger.info("Extraído ZIP → %s", dest_dir.name)
            return True

        tar_exts = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
        if any(suffix_chain.endswith(ext) for ext in tar_exts):
            with tarfile.open(archive) as tf:
                tf.extractall(dest_dir)
            logger.info("Extraído TAR → %s", dest_dir.name)
            return True

    except zipfile.BadZipFile as exc:
        logger.error("ZIP corrupto %s: %s", archive.name, exc)
    except tarfile.TarError as exc:
        logger.error("TAR corrupto %s: %s", archive.name, exc)
    except OSError as exc:
        logger.error("Error de sistema al extraer %s: %s", archive.name, exc)

    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — REGISTRO CONSOLIDADO DE DESCARGAS
# ══════════════════════════════════════════════════════════════════════════════

class Registry:
    """
    Índice central de todos los archivos descargados.

    Cada entrada mapea:
      id, source, dataset_name, filename, local_path, source_url,
      file_type, categoria_proyecto, label, annotation_file,
      md5, size_bytes, downloaded_at

    Al finalizar, se persiste en CSV y JSON para uso inmediato
    en pipelines de entrenamiento (p. ej. pandas, PyTorch Dataset, etc.).
    """

    # Orden y nombre de columnas en el CSV/JSON de salida
    FIELDS = [
        "id",
        "source",
        "dataset_name",
        "filename",
        "local_path",
        "source_url",
        "file_type",           # "image" | "annotation" | "archive" | "other"
        "categoria_proyecto",  # "Aprobado" | "No Aprobado"
        "label",               # etiqueta semántica o nombre de carpeta
        "annotation_file",     # ruta del archivo de anotación asociado (si existe)
        "md5",
        "size_bytes",
        "downloaded_at",
    ]

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        # Conjunto de rutas absolutas ya registradas (evita duplicados)
        self._registered_paths: Set[str] = set()

    def add(
        self,
        *,
        source: str,
        dataset_name: str,
        filename: str,
        local_path: Path,
        source_url: str,
        file_type: str,
        categoria_proyecto: str,
        label: str = "",
        annotation_file: str = "",
    ) -> None:
        """
        Agrega un archivo al registro. Si la ruta ya fue registrada,
        la omite para evitar entradas duplicadas.
        """
        abs_path = str(local_path.resolve())
        if abs_path in self._registered_paths:
            return
        self._registered_paths.add(abs_path)

        size     = local_path.stat().st_size if local_path.exists() else 0
        checksum = md5_checksum(local_path)  if local_path.exists() else ""

        self.records.append({
            "id":                  str(uuid.uuid4()),
            "source":              source,
            "dataset_name":        dataset_name,
            "filename":            filename,
            "local_path":          abs_path,
            "source_url":          source_url,
            "file_type":           file_type,
            "categoria_proyecto":  categoria_proyecto,
            "label":               label,
            "annotation_file":     annotation_file,
            "md5":                 checksum,
            "size_bytes":          size,
            "downloaded_at":       datetime.now(timezone.utc).isoformat(),
        })

    def register_directory(
        self,
        directory: Path,
        source: str,
        dataset_name: str,
        source_url: str,
        categoria_proyecto: str,
    ) -> None:
        """
        Escanea recursivamente `directory` y registra todos los archivos
        de imagen y anotación que encuentra. Usa el nombre de la carpeta
        inmediata como etiqueta (común en datasets organizados por clase).
        """
        for fpath in sorted(directory.rglob("*")):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            if ext in IMAGE_EXTS:
                ftype = "image"
            elif ext in ANNOTATION_EXTS:
                ftype = "annotation"
            else:
                continue
            self.add(
                source=source,
                dataset_name=dataset_name,
                filename=fpath.name,
                local_path=fpath,
                source_url=source_url,
                file_type=ftype,
                categoria_proyecto=categoria_proyecto,
                label=fpath.parent.name,
            )

    def save(self, logger: logging.Logger) -> None:
        """
        Guarda el registro en dos formatos:
          • CSV  → fácil de abrir en Excel o pandas
          • JSON → útil para pipelines Python y APIs
        """
        csv_path  = BASE_DIR / "download_registry.csv"
        json_path = BASE_DIR / "download_registry.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.records)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.records, fh, indent=2, ensure_ascii=False)

        logger.info(
            "Registro guardado — %d entradas:\n  → %s\n  → %s",
            len(self.records), csv_path.resolve(), json_path.resolve(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — DESCARGADOR 1: ROBOFLOW UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

def download_roboflow(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga el dataset 'pavement-crack-detection-r4n7n' desde Roboflow.

    Flujo:
      1. Solicita a la API de Roboflow una URL firmada de exportación ZIP.
      2. Descarga el ZIP.
      3. Extrae el contenido.
      4. Registra cada imagen y anotación con categoría "Aprobado".

    Requiere: variable de entorno ROBOFLOW_API_KEY
    Docs:     https://docs.roboflow.com/api-reference/images/get-images
    """
    logger.info("═══ [1/5] ROBOFLOW ═══════════════════════════════════")
    out_dir    = SOURCE_DIRS["roboflow"]
    source_url = f"https://universe.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ROBOFLOW_API_KEY:
        logger.warning(
            "ROBOFLOW_API_KEY no está definida — fuente omitida.\n"
            "  1. Crea una cuenta gratuita en https://app.roboflow.com/\n"
            "  2. Ve a Settings → API Key\n"
            "  3. Ejecuta:  export ROBOFLOW_API_KEY=<tu_clave>"
        )
        return

    session = build_session()

    # Paso 1 — Pedir la URL de exportación en el formato seleccionado
    export_endpoint = (
        f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
        f"/{ROBOFLOW_VERSION}/{ROBOFLOW_FORMAT}"
    )
    logger.info("Solicitando exportación Roboflow (formato: %s) …", ROBOFLOW_FORMAT)
    try:
        resp = session.get(
            export_endpoint,
            params={"api_key": ROBOFLOW_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as exc:
        logger.error(
            "HTTP %s en Roboflow API.\n"
            "  Verifica: workspace='%s', project='%s', version=%d, formato='%s'",
            exc.response.status_code,
            ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION, ROBOFLOW_FORMAT,
        )
        return
    except Exception as exc:
        logger.error("Error contactando Roboflow: %s", exc)
        return

    # La respuesta tiene la forma: {"export": {"link": "<url_firmada>"}}
    export_link = data.get("export", {}).get("link", "")
    if not export_link:
        logger.error(
            "Roboflow no devolvió enlace de descarga.\n"
            "Respuesta recibida: %s\n"
            "Tip: el formato '%s' puede no estar disponible para este dataset.",
            json.dumps(data, indent=2)[:400],
            ROBOFLOW_FORMAT,
        )
        return

    # Paso 2 — Descargar el ZIP
    zip_path = out_dir / f"roboflow_{ROBOFLOW_FORMAT}_v{ROBOFLOW_VERSION}.zip"
    logger.info("Descargando ZIP de Roboflow …")
    if not download_file(export_link, zip_path, session, logger):
        return

    # Paso 3 — Extraer
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    if not extract_archive(zip_path, extract_dir, logger):
        return

    # Paso 4 — Registrar archivos extraídos como "Aprobado"
    registry.register_directory(
        extract_dir, "roboflow", ROBOFLOW_PROJECT,
        source_url, CAT_APROBADO,
    )
    logger.info("Roboflow: completado → %s", extract_dir)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — DESCARGADOR 2: MENDELEY DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_mendeley(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga el dataset 88kdyyc73h desde Mendeley Data.

    Flujo:
      1. Consulta la API pública de Mendeley Data para listar los archivos
         del dataset (no requiere autenticación).
      2. Descarga cada archivo.
      3. Extrae automáticamente los archivos comprimidos.
      4. Registra todo como "Aprobado".

    Docs: https://data.mendeley.com/api-documentation
    """
    logger.info("═══ [2/5] MENDELEY DATA ══════════════════════════════")
    out_dir     = SOURCE_DIRS["mendeley"]
    dataset_url = f"https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/{MENDELEY_VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    # La API pública de Mendeley ha cambiado de URL en el pasado;
    # intentamos dos endpoints por si uno falla.
    candidate_endpoints = [
        (
            f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
            f"/versions/{MENDELEY_VERSION}/files"
        ),
        (
            f"https://data.mendeley.com/api/datasets-v2/datasets/{MENDELEY_DATASET_ID}"
            f"/versions/{MENDELEY_VERSION}/files"
        ),
    ]

    files: list = []
    for url in candidate_endpoints:
        logger.info("Probando endpoint Mendeley: %s", url)
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            # La API puede devolver una lista o un dict con clave 'files'/'data'
            if isinstance(payload, list):
                files = payload
            elif isinstance(payload, dict):
                files = payload.get("files", payload.get("data", []))
            if files:
                logger.info("Mendeley: %d archivo(s) encontrado(s)", len(files))
                break
        except requests.HTTPError as exc:
            logger.debug("HTTP %s en %s", exc.response.status_code, url)
        except Exception as exc:
            logger.debug("Error en %s: %s", url, exc)

    if not files:
        logger.error(
            "No se pudo obtener la lista de archivos de Mendeley Data.\n"
            "  Dataset ID: %s  |  Versión: %d\n"
            "  Descarga manual: %s",
            MENDELEY_DATASET_ID, MENDELEY_VERSION, dataset_url,
        )
        return

    for item in files:
        # Los nombres de campo pueden variar entre versiones de la API
        filename = (
            item.get("filename")
            or item.get("name")
            or f"mendeley_{uuid.uuid4().hex[:8]}"
        )
        dl_url = (
            item.get("content_details", {}).get("download_url")
            or item.get("download_url")
            or item.get("fileUrl")
            or ""
        )
        if not dl_url:
            logger.warning("Sin URL de descarga para: %s", filename)
            continue

        dest = unique_path(out_dir, filename)
        logger.info("Descargando: %s", filename)
        if not download_file(dl_url, dest, session, logger):
            continue

        ext = dest.suffix.lower()
        if ext in IMAGE_EXTS:
            ftype = "image"
        elif ext in ANNOTATION_EXTS:
            ftype = "annotation"
        elif ext in ARCHIVE_EXTS:
            ftype = "archive"
            # Extraer y registrar el contenido del comprimido
            extract_dir = out_dir / dest.stem
            extract_dir.mkdir(exist_ok=True)
            if extract_archive(dest, extract_dir, logger):
                registry.register_directory(
                    extract_dir, "mendeley",
                    f"mendeley-{MENDELEY_DATASET_ID}",
                    dataset_url, CAT_APROBADO,
                )
        else:
            ftype = "other"

        registry.add(
            source="mendeley",
            dataset_name=f"mendeley-{MENDELEY_DATASET_ID}",
            filename=dest.name,
            local_path=dest,
            source_url=dl_url,
            file_type=ftype,
            categoria_proyecto=CAT_APROBADO,
        )

    logger.info("Mendeley: completado.")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — DESCARGADOR 3: GITHUB
# ══════════════════════════════════════════════════════════════════════════════

def download_github(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga imágenes y anotaciones del repositorio juhuyan/CrackDataset_DL_HY.

    Flujo:
      1. Obtiene la rama por defecto del repositorio.
      2. Descarga el árbol completo de archivos con la API de GitHub
         (una sola petición, sin importar cuántos archivos haya).
      3. Filtra los archivos por extensión (imágenes y anotaciones).
      4. Descarga cada archivo desde raw.githubusercontent.com
         preservando la estructura de carpetas del repositorio.
      5. Registra todo como "Aprobado".

    Nota: sin GITHUB_TOKEN el límite es 60 peticiones/hora.
          Con token (gratuito) sube a 5 000 peticiones/hora.
    """
    logger.info("═══ [3/5] GITHUB ════════════════════════════════════")
    out_dir  = SOURCE_DIRS["github"]
    repo_url = f"https://github.com/{GITHUB_REPO}"
    out_dir.mkdir(parents=True, exist_ok=True)

    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    session = build_session(headers)

    # Obtener la rama por defecto (puede ser 'main' o 'master')
    repo_info_url = f"https://api.github.com/repos/{GITHUB_REPO}"
    try:
        resp = session.get(repo_info_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        default_branch = resp.json().get("default_branch", "main")
    except Exception as exc:
        logger.warning("No se pudo obtener la rama por defecto (%s); usando 'main'", exc)
        default_branch = "main"

    # Una sola petición trae el árbol completo del repositorio
    tree_url = (
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/git/trees/{default_branch}?recursive=1"
    )
    logger.info("Obteniendo árbol del repositorio (rama: %s) …", default_branch)
    try:
        resp = session.get(tree_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        tree_data = resp.json()
    except requests.HTTPError as exc:
        logger.error(
            "HTTP %s al acceder al árbol de GitHub.\n"
            "  Si el repositorio es privado, define GITHUB_TOKEN.\n"
            "  URL: %s",
            exc.response.status_code, tree_url,
        )
        return
    except Exception as exc:
        logger.error("Error al acceder a GitHub API: %s", exc)
        return

    if tree_data.get("truncated"):
        logger.warning(
            "El árbol de GitHub fue truncado (repo muy grande). "
            "Algunos archivos podrían quedar sin descargar."
        )

    all_blobs = [item for item in tree_data.get("tree", []) if item["type"] == "blob"]
    targets   = [
        b for b in all_blobs
        if Path(b["path"]).suffix.lower() in IMAGE_EXTS | ANNOTATION_EXTS
    ]
    logger.info(
        "Repositorio: %d archivo(s) totales → %d imágenes/anotaciones",
        len(all_blobs), len(targets),
    )

    for item in targets:
        rel_path = item["path"]
        raw_url  = (
            f"https://raw.githubusercontent.com/{GITHUB_REPO}"
            f"/{default_branch}/{rel_path}"
        )
        # Preservar la estructura de carpetas original del repositorio
        dest = out_dir / rel_path
        if dest.exists():
            dest = unique_path(dest.parent, dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info("  ↓ %s", rel_path)
        if not download_file(raw_url, dest, session, logger):
            continue

        ext   = Path(rel_path).suffix.lower()
        ftype = "image" if ext in IMAGE_EXTS else "annotation"
        label = Path(rel_path).parent.name

        registry.add(
            source="github",
            dataset_name="CrackDataset_DL_HY",
            filename=dest.name,
            local_path=dest,
            source_url=raw_url,
            file_type=ftype,
            categoria_proyecto=CAT_APROBADO,
            label=label,
        )
        # Pausa mínima para no consumir el rate limit innecesariamente
        time.sleep(0.05)

    logger.info("GitHub: completado.")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — DESCARGADOR 4: FIGSHARE
# ══════════════════════════════════════════════════════════════════════════════

def download_figshare(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga el artículo 25103138 desde Figshare.

    Flujo:
      1. Consulta la API pública v2 de Figshare para listar archivos
         (no requiere autenticación).
      2. Descarga cada archivo.
      3. Extrae automáticamente los comprimidos encontrados.
      4. Registra todo como "Aprobado".

    Docs: https://docs.figshare.com/#article_files_list
    """
    logger.info("═══ [4/5] FIGSHARE ══════════════════════════════════")
    out_dir     = SOURCE_DIRS["figshare"]
    article_url = (
        f"https://figshare.com/articles/dataset/"
        f"Pavement_cracks_from_UAV_imagery-1388/{FIGSHARE_ARTICLE_ID}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    session  = build_session()
    api_url  = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"

    logger.info("Listando archivos en Figshare …")
    try:
        resp = session.get(api_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        files = resp.json()
    except Exception as exc:
        logger.error("Error al acceder a Figshare API: %s", exc)
        return

    if not isinstance(files, list):
        logger.error("Respuesta inesperada de Figshare: %s", str(files)[:200])
        return

    logger.info("Figshare: %d archivo(s) encontrado(s)", len(files))

    for f in files:
        filename = f.get("name", f"figshare_{f.get('id', uuid.uuid4().hex)}")
        dl_url   = f.get("download_url", "")
        size_b   = f.get("size", 0)

        if not dl_url:
            logger.warning("Sin URL de descarga para: %s", filename)
            continue

        dest = unique_path(out_dir, filename)
        logger.info("Descargando: %s  (%s)", filename, human_size(size_b))
        if not download_file(dl_url, dest, session, logger):
            continue

        ext = dest.suffix.lower()
        if ext in IMAGE_EXTS:
            ftype = "image"
        elif ext in ANNOTATION_EXTS:
            ftype = "annotation"
        elif ext in ARCHIVE_EXTS:
            ftype = "archive"
            extract_dir = out_dir / dest.stem
            extract_dir.mkdir(exist_ok=True)
            if extract_archive(dest, extract_dir, logger):
                registry.register_directory(
                    extract_dir, "figshare",
                    f"figshare-{FIGSHARE_ARTICLE_ID}",
                    article_url, CAT_APROBADO,
                )
        else:
            ftype = "other"

        registry.add(
            source="figshare",
            dataset_name=f"figshare-{FIGSHARE_ARTICLE_ID}",
            filename=dest.name,
            local_path=dest,
            source_url=dl_url,
            file_type=ftype,
            categoria_proyecto=CAT_APROBADO,
        )

    logger.info("Figshare: completado.")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — DESCARGADOR 5: COCO val2017  (categoría "No Aprobado")
# ══════════════════════════════════════════════════════════════════════════════

def download_coco(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga un subconjunto de imágenes del COCO val2017 como ejemplos
    negativos ("No Aprobado") para el clasificador.

    El conjunto val2017 tiene exactamente 5 000 imágenes de escenas y
    objetos cotidianos (personas, autos, animales, muebles, comida, etc.)
    sin ninguna categoría relacionada con grietas o pavimento dañado.
    Son ideales para enseñarle al modelo a rechazar fotos que no son reportes
    válidos de grietas.

    Flujo:
      1. Descarga el ZIP de anotaciones de COCO (annotations_trainval2017.zip).
      2. Extrae el archivo instances_val2017.json.
      3. Parsea la lista de imágenes del set de validación.
      4. Selecciona aleatoriamente hasta COCO_MAX_IMAGES imágenes (semilla fija
         para reproducibilidad).
      5. Descarga cada imagen desde images.cocodataset.org/val2017/.
      6. Guarda un JSON de anotaciones compacto por imagen.
      7. Registra todo como "No Aprobado".

    URLs oficiales de COCO: https://cocodataset.org/#download
    """
    logger.info("═══ [5/5] COCO val2017 (No Aprobado) ════════════════")
    out_dir      = SOURCE_DIRS["coco_control"]
    images_dir   = out_dir / "images"
    annots_dir   = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annots_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    # ── Paso 1: descargar el ZIP de anotaciones ────────────────────────────
    annot_zip = out_dir / "annotations_trainval2017.zip"
    if not annot_zip.exists():
        logger.info(
            "Descargando anotaciones COCO (~240 MB) …\n"
            "  Fuente: %s", COCO_ANNOTATIONS_URL,
        )
        if not download_file(COCO_ANNOTATIONS_URL, annot_zip, session, logger):
            logger.error(
                "No se pudo descargar las anotaciones de COCO.\n"
                "  Descarga manual: %s", COCO_ANNOTATIONS_URL,
            )
            return
    else:
        logger.info("ZIP de anotaciones ya existe — reutilizando.")

    # ── Paso 2: extraer solo instances_val2017.json ────────────────────────
    annot_json_path = annots_dir / "instances_val2017.json"
    if not annot_json_path.exists():
        logger.info("Extrayendo instances_val2017.json del ZIP …")
        try:
            with zipfile.ZipFile(annot_zip, "r") as zf:
                target = "annotations/instances_val2017.json"
                if target not in zf.namelist():
                    logger.error(
                        "'%s' no encontrado en el ZIP. "
                        "Archivos disponibles: %s",
                        target, zf.namelist()[:10],
                    )
                    return
                # Extraer directamente a annots_dir sin estructura de carpetas
                data = zf.read(target)
                annot_json_path.write_bytes(data)
            logger.info("Anotaciones extraídas → %s", annot_json_path.name)
        except (zipfile.BadZipFile, KeyError) as exc:
            logger.error("Error al extraer anotaciones COCO: %s", exc)
            return

    # ── Paso 3: parsear la lista de imágenes ─────────────────────────────
    logger.info("Parseando instances_val2017.json …")
    try:
        with open(annot_json_path, encoding="utf-8") as fh:
            coco_data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error al leer el JSON de COCO: %s", exc)
        return

    all_images: List[Dict[str, Any]] = coco_data.get("images", [])
    all_annots: List[Dict[str, Any]] = coco_data.get("annotations", [])
    categories: List[Dict[str, Any]] = coco_data.get("categories", [])

    # Mapa rápido: image_id → lista de anotaciones
    annots_by_image: Dict[int, List[Dict]] = {}
    for ann in all_annots:
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    # Mapa rápido: category_id → nombre
    cat_names: Dict[int, str] = {c["id"]: c["name"] for c in categories}

    logger.info(
        "COCO val2017: %d imágenes totales, %d anotaciones, %d categorías",
        len(all_images), len(all_annots), len(categories),
    )

    # ── Paso 4: selección aleatoria reproducible ──────────────────────────
    rng = random.Random(COCO_RANDOM_SEED)
    selected = list(all_images)
    rng.shuffle(selected)
    if COCO_MAX_IMAGES and len(selected) > COCO_MAX_IMAGES:
        selected = selected[:COCO_MAX_IMAGES]
        logger.info(
            "Seleccionadas %d/%d imágenes (semilla=%d)",
            len(selected), len(all_images), COCO_RANDOM_SEED,
        )

    # ── Pasos 5-7: descargar imágenes y guardar anotaciones ───────────────
    ok_count   = 0
    fail_count = 0

    for idx, img_meta in enumerate(selected, start=1):
        img_id   = img_meta["id"]
        img_file = img_meta["file_name"]               # ej. 000000001000.jpg
        coco_url = COCO_IMAGE_BASE_URL + img_file

        # Nombre único con prefijo para evitar colisión con otras fuentes
        local_name = f"coco_{img_file}"
        dest_img   = unique_path(images_dir, local_name)

        if idx % 200 == 0 or idx == 1:
            logger.info(
                "  COCO [%d/%d] descargando: %s", idx, len(selected), img_file
            )

        if not download_file(coco_url, dest_img, session, logger):
            fail_count += 1
            continue

        # Crear un JSON de anotación compacto por imagen
        img_annotations = annots_by_image.get(img_id, [])
        compact_annot = {
            "image_id":   img_id,
            "file_name":  img_file,
            "width":      img_meta.get("width"),
            "height":     img_meta.get("height"),
            "coco_url":   img_meta.get("coco_url", coco_url),
            "annotations": [
                {
                    "id":          a["id"],
                    "category_id": a["category_id"],
                    "category":    cat_names.get(a["category_id"], "unknown"),
                    "bbox":        a.get("bbox"),
                    "area":        a.get("area"),
                    "iscrowd":     a.get("iscrowd", 0),
                }
                for a in img_annotations
            ],
        }
        annot_file = annots_dir / f"coco_{Path(img_file).stem}.json"
        try:
            annot_file.write_text(
                json.dumps(compact_annot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("No se pudo guardar anotación de %s: %s", img_file, exc)
            annot_file = Path("")

        # Etiquetas COCO presentes en la imagen (para el registro)
        coco_labels = ", ".join(
            sorted({cat_names.get(a["category_id"], "") for a in img_annotations})
        )

        registry.add(
            source="coco",
            dataset_name="coco_val2017",
            filename=dest_img.name,
            local_path=dest_img,
            source_url=coco_url,
            file_type="image",
            categoria_proyecto=CAT_NO_APROBADO,
            label=coco_labels or "sin_etiqueta",
            annotation_file=str(annot_file.resolve()) if annot_file.name else "",
        )

        # También registrar el archivo de anotación
        if annot_file.exists():
            registry.add(
                source="coco",
                dataset_name="coco_val2017",
                filename=annot_file.name,
                local_path=annot_file,
                source_url=coco_url,
                file_type="annotation",
                categoria_proyecto=CAT_NO_APROBADO,
                label=coco_labels or "sin_etiqueta",
            )

        ok_count += 1
        # Pausa breve para no saturar el servidor de COCO
        time.sleep(0.02)

    logger.info(
        "COCO: completado — %d descargadas, %d fallidas.", ok_count, fail_count
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — PUNTO DE ENTRADA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orquesta la descarga de todas las fuentes en orden y guarda el registro.
    Cada fuente se ejecuta dentro de un bloque try/except independiente para
    que un fallo en una fuente no detenga las demás.
    """
    logger   = setup_logging()
    registry = Registry()

    logger.info(
        "Dataset Builder — Clasificador de Grietas en Calles de Toluca\n"
        "  Inicio: %s\n"
        "  Directorio de salida: %s",
        datetime.now(timezone.utc).isoformat(),
        BASE_DIR.resolve(),
    )

    # Crear todas las carpetas antes de iniciar las descargas
    for d in SOURCE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Pipeline de descarga: (nombre_fuente, función_descargadora)
    pipeline = [
        ("roboflow",     download_roboflow),
        ("mendeley",     download_mendeley),
        ("github",       download_github),
        ("figshare",     download_figshare),
        ("coco_control", download_coco),
    ]

    errors: List[Tuple[str, str]] = []

    for source_name, downloader_fn in pipeline:
        try:
            downloader_fn(registry, logger)
        except KeyboardInterrupt:
            logger.warning("Descarga interrumpida por el usuario (Ctrl+C).")
            break
        except Exception as exc:
            logger.exception("Error inesperado en [%s]: %s", source_name, exc)
            errors.append((source_name, str(exc)))

    # Guardar el registro consolidado
    registry.save(logger)

    # ── Resumen final ─────────────────────────────────────────────────────
    total = len(registry.records)
    logger.info("═" * 62)
    logger.info("RESUMEN FINAL")
    logger.info("  Total de archivos registrados: %d", total)
    logger.info("")

    # Resumen por fuente
    logger.info("  Por fuente:")
    all_sources = list(SOURCE_DIRS.keys()) + ["coco"]
    for src in all_sources:
        n     = sum(1 for r in registry.records if r["source"] == src)
        imgs  = sum(1 for r in registry.records if r["source"] == src and r["file_type"] == "image")
        annos = sum(1 for r in registry.records if r["source"] == src and r["file_type"] == "annotation")
        if n:
            logger.info("    %-15s  %4d archivos  (%d imgs + %d anots)", src, n, imgs, annos)

    # Resumen por categoría del proyecto
    logger.info("")
    logger.info("  Por categoría del proyecto:")
    for cat in [CAT_APROBADO, CAT_NO_APROBADO]:
        n_imgs = sum(
            1 for r in registry.records
            if r["categoria_proyecto"] == cat and r["file_type"] == "image"
        )
        logger.info("    %-15s  %d imágenes", cat, n_imgs)

    if errors:
        logger.warning("")
        logger.warning("  Fuentes con errores: %s", [e[0] for e in errors])

    logger.info("")
    logger.info("  Registro: %s", (BASE_DIR / "download_registry.csv").resolve())
    logger.info("  Log:      %s", (BASE_DIR / "download.log").resolve())
    logger.info("═" * 62)
    logger.info("Listo.")


if __name__ == "__main__":
    main()
