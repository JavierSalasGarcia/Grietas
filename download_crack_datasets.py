#!/usr/bin/env python3
"""
Pavement Crack Dataset Downloader
==================================
Descarga imágenes y anotaciones de cuatro repositorios públicos:
  1. Roboflow Universe  – pavement-crack-detection-r4n7n
  2. Mendeley Data      – dataset 88kdyyc73h (v1)
  3. GitHub             – juhuyan/CrackDataset_DL_HY
  4. Figshare           – article 25103138

Uso rápido
----------
    pip install requests
    export ROBOFLOW_API_KEY=<tu_clave>   # necesaria solo para Roboflow
    export GITHUB_TOKEN=<tu_token>       # opcional, sube el límite a 5 000 req/h
    python download_crack_datasets.py

Los archivos se guardan en ./pavement_crack_datasets/<fuente>/ y se genera
un registro en CSV y JSON con metadatos de cada descarga.
"""

import csv
import hashlib
import json
import logging
import os
import sys
import tarfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import unquote, urlparse

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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path("pavement_crack_datasets")

SOURCE_DIRS: Dict[str, Path] = {
    "roboflow": BASE_DIR / "01_roboflow",
    "mendeley": BASE_DIR / "02_mendeley",
    "github":   BASE_DIR / "03_github",
    "figshare": BASE_DIR / "04_figshare",
}

# ── Roboflow ──────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = "mcmaster-vkq5k"
ROBOFLOW_PROJECT   = "pavement-crack-detection-r4n7n"
ROBOFLOW_VERSION   = 1
# Opciones de formato: yolov8, yolov5, coco, pascal-voc, tensorflow, etc.
ROBOFLOW_FORMAT    = "yolov8"

# ── Mendeley Data ─────────────────────────────────────────────────────────────
MENDELEY_DATASET_ID = "88kdyyc73h"
MENDELEY_VERSION    = 1

# ── GitHub ────────────────────────────────────────────────────────────────────
GITHUB_REPO  = "juhuyan/CrackDataset_DL_HY"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # sube el límite de rate a 5 000 req/h

# ── Figshare ──────────────────────────────────────────────────────────────────
FIGSHARE_ARTICLE_ID = 25103138

# ── HTTP ──────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 120          # segundos
RETRY_ATTEMPTS   = 4
RETRY_BACKOFF    = 2.0          # segundos entre reintentos (crece exponencialmente)
CHUNK_SIZE       = 64 * 1024    # 64 KB por chunk al descargar

# Tamaño máximo de archivo a descargar (0 = sin límite)
MAX_FILE_BYTES: int = 0

# ── Extensiones reconocidas ───────────────────────────────────────────────────
IMAGE_EXTS: Set[str]      = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_EXTS: Set[str] = {".xml", ".json", ".txt", ".csv", ".yaml", ".yml"}
ARCHIVE_EXTS: Set[str]    = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = BASE_DIR / "download.log"

    logger = logging.getLogger("crack_dl")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# SESIÓN HTTP CON REINTENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def build_session(extra_headers: Dict[str, str] | None = None) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=RETRY_ATTEMPTS,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "PavementCrackDownloader/1.0"})
    if extra_headers:
        session.headers.update(extra_headers)
    return session


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def unique_path(directory: Path, filename: str) -> Path:
    """Devuelve una ruta que no existe; añade _N al stem si ya hay colisión."""
    stem   = Path(filename).stem
    suffix = Path(filename).suffix
    dest   = directory / filename
    n = 1
    while dest.exists():
        dest = directory / f"{stem}_{n}{suffix}"
        n += 1
    return dest


def md5_checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    session: requests.Session,
    logger: logging.Logger,
) -> bool:
    """
    Descarga *url* a *dest* en modo streaming.
    Devuelve True si la descarga fue exitosa.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()

            content_length = int(resp.headers.get("Content-Length", 0))
            if MAX_FILE_BYTES and content_length > MAX_FILE_BYTES:
                logger.warning(
                    "Omitido (tamaño %d B > límite %d B): %s",
                    content_length, MAX_FILE_BYTES, url,
                )
                return False

            with open(dest, "wb") as fh:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

        logger.debug("OK  %s  (%d B) → %s", url, downloaded, dest)
        return True

    except requests.HTTPError as exc:
        logger.error("HTTP %s al descargar %s", exc.response.status_code, url)
    except requests.ConnectionError:
        logger.error("Error de conexión: %s", url)
    except requests.Timeout:
        logger.error("Timeout: %s", url)
    except OSError as exc:
        logger.error("Error de escritura en %s: %s", dest, exc)

    if dest.exists():
        dest.unlink(missing_ok=True)
    return False


def extract_archive(archive: Path, dest_dir: Path, logger: logging.Logger) -> bool:
    """Extrae un zip o tar en dest_dir. Devuelve True si tuvo éxito."""
    suffix = "".join(archive.suffixes).lower()
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest_dir)
            logger.info("Extraído ZIP: %s", archive.name)
            return True
        if any(suffix.endswith(ext) for ext in (".tar", ".gz", ".tgz", ".bz2", ".xz")):
            with tarfile.open(archive) as tf:
                tf.extractall(dest_dir)
            logger.info("Extraído TAR: %s", archive.name)
            return True
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        logger.error("No se pudo extraer %s: %s", archive.name, exc)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE DESCARGAS
# ═══════════════════════════════════════════════════════════════════════════════

class Registry:
    """Acumula metadatos de cada archivo descargado y los persiste en CSV + JSON."""

    FIELDS = [
        "id", "source", "dataset_name", "filename",
        "local_path", "source_url", "file_type",
        "label", "annotation_file",
        "md5", "size_bytes", "downloaded_at",
    ]

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._known_paths: Set[str] = set()

    def add(
        self, *,
        source: str,
        dataset_name: str,
        filename: str,
        local_path: Path,
        source_url: str,
        file_type: str,          # "image" | "annotation" | "archive" | "other"
        label: str = "",
        annotation_file: str = "",
    ) -> None:
        abs_path = str(local_path.resolve())
        if abs_path in self._known_paths:
            return
        self._known_paths.add(abs_path)

        size     = local_path.stat().st_size if local_path.exists() else 0
        checksum = md5_checksum(local_path)  if local_path.exists() else ""

        self.records.append({
            "id":              str(uuid.uuid4()),
            "source":          source,
            "dataset_name":    dataset_name,
            "filename":        filename,
            "local_path":      abs_path,
            "source_url":      source_url,
            "file_type":       file_type,
            "label":           label,
            "annotation_file": annotation_file,
            "md5":             checksum,
            "size_bytes":      size,
            "downloaded_at":   datetime.now(timezone.utc).isoformat(),
        })

    def register_directory(
        self,
        directory: Path,
        source: str,
        dataset_name: str,
        source_url: str,
    ) -> None:
        """Registra todos los archivos de imagen/anotación en *directory*."""
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
                label=fpath.parent.name,
            )

    def save(self, logger: logging.Logger) -> None:
        csv_path  = BASE_DIR / "download_registry.csv"
        json_path = BASE_DIR / "download_registry.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.records)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.records, fh, indent=2, ensure_ascii=False)

        logger.info(
            "Registro guardado (%d entradas):\n  %s\n  %s",
            len(self.records), csv_path, json_path,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ROBOFLOW UNIVERSE
#    URL: https://universe.roboflow.com/mcmaster-vkq5k/pavement-crack-detection-r4n7n
# ═══════════════════════════════════════════════════════════════════════════════

def download_roboflow(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga el dataset a través de la API REST de Roboflow.
    Requiere ROBOFLOW_API_KEY (variable de entorno).

    Referencia: https://docs.roboflow.com/api-reference/images/get-images
    """
    logger.info("═══ [1/4] ROBOFLOW ════════════════════════════════════")
    out_dir = SOURCE_DIRS["roboflow"]
    out_dir.mkdir(parents=True, exist_ok=True)
    source_url = (
        f"https://universe.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
    )

    if not ROBOFLOW_API_KEY:
        logger.warning(
            "ROBOFLOW_API_KEY no definida → fuente omitida.\n"
            "  Obtén tu clave gratis en https://app.roboflow.com/\n"
            "  Luego ejecuta:  export ROBOFLOW_API_KEY=<tu_clave>"
        )
        return

    session = build_session()

    # Solicitar URL de exportación del dataset
    export_endpoint = (
        f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
        f"/{ROBOFLOW_VERSION}/{ROBOFLOW_FORMAT}"
    )
    logger.info("Solicitando exportación: %s (formato: %s)", export_endpoint, ROBOFLOW_FORMAT)
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
            "Error HTTP %s en Roboflow. Verifica workspace, project y API key.",
            exc.response.status_code,
        )
        return
    except Exception as exc:
        logger.error("Error al contactar Roboflow API: %s", exc)
        return

    # La respuesta contiene {"export": {"link": "<url_firmada>"}}
    export_link = data.get("export", {}).get("link", "")
    if not export_link:
        logger.error(
            "Roboflow no devolvió enlace de descarga.\n"
            "Respuesta: %s\n"
            "Verifica que el formato '%s' esté disponible para este dataset.",
            json.dumps(data, indent=2)[:500],
            ROBOFLOW_FORMAT,
        )
        return

    # Descargar el ZIP del dataset
    zip_path = out_dir / f"roboflow_{ROBOFLOW_FORMAT}_v{ROBOFLOW_VERSION}.zip"
    logger.info("Descargando archivo ZIP de Roboflow …")
    if not download_file(export_link, zip_path, session, logger):
        return

    # Extraer
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    if not extract_archive(zip_path, extract_dir, logger):
        return

    # Registrar archivos extraídos
    registry.register_directory(extract_dir, "roboflow", ROBOFLOW_PROJECT, source_url)
    logger.info(
        "Roboflow: completado. Archivos en %s",
        extract_dir.relative_to(Path.cwd()),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MENDELEY DATA
#    URL: https://data.mendeley.com/datasets/88kdyyc73h/1
# ═══════════════════════════════════════════════════════════════════════════════

def download_mendeley(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga desde Mendeley Data a través de su API pública (sin autenticación).
    Documentación: https://data.mendeley.com/api-documentation
    """
    logger.info("═══ [2/4] MENDELEY DATA ══════════════════════════════")
    out_dir = SOURCE_DIRS["mendeley"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_url = (
        f"https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/{MENDELEY_VERSION}"
    )

    session = build_session()

    # Intentar con la API v2 pública de Mendeley Data
    endpoints_to_try = [
        (
            f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
            f"/versions/{MENDELEY_VERSION}/files"
        ),
        # Alternativa si la URL anterior cambia
        (
            f"https://data.mendeley.com/api/datasets-v2/datasets/{MENDELEY_DATASET_ID}"
            f"/versions/{MENDELEY_VERSION}/files"
        ),
    ]

    files: list = []
    for url in endpoints_to_try:
        logger.info("Probando endpoint Mendeley: %s", url)
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            # La API puede devolver {"files": [...]} o directamente [...]
            if isinstance(payload, list):
                files = payload
            elif isinstance(payload, dict):
                files = payload.get("files", payload.get("data", []))
            if files:
                logger.info("Encontrados %d archivo(s) en Mendeley", len(files))
                break
        except requests.HTTPError as exc:
            logger.debug("HTTP %s en %s", exc.response.status_code, url)
        except Exception as exc:
            logger.debug("Error en %s: %s", url, exc)

    if not files:
        logger.error(
            "No se pudo obtener la lista de archivos de Mendeley Data.\n"
            "  Dataset: %s\n"
            "  Descarga manual disponible en: %s",
            MENDELEY_DATASET_ID, dataset_url,
        )
        return

    for item in files:
        # Los campos varían según la versión de la API
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
            # Extraer y registrar el contenido del archivo comprimido
            extract_dir = out_dir / dest.stem
            extract_dir.mkdir(exist_ok=True)
            if extract_archive(dest, extract_dir, logger):
                registry.register_directory(
                    extract_dir, "mendeley",
                    f"mendeley-{MENDELEY_DATASET_ID}", dataset_url,
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
        )

    logger.info("Mendeley: completado.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GITHUB
#    URL: https://github.com/juhuyan/CrackDataset_DL_HY
# ═══════════════════════════════════════════════════════════════════════════════

def download_github(registry: Registry, logger: logging.Logger) -> None:
    """
    Recorre el árbol de archivos del repositorio y descarga
    imágenes y anotaciones usando la API REST de GitHub.
    Con GITHUB_TOKEN el límite de peticiones sube de 60 a 5 000 por hora.
    """
    logger.info("═══ [3/4] GITHUB ════════════════════════════════════")
    out_dir = SOURCE_DIRS["github"]
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{GITHUB_REPO}"

    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    session = build_session(headers)

    # Obtener la rama por defecto del repositorio
    repo_info_url = f"https://api.github.com/repos/{GITHUB_REPO}"
    try:
        resp = session.get(repo_info_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        default_branch = resp.json().get("default_branch", "main")
    except Exception as exc:
        logger.warning("No se pudo obtener info del repo (%s); usando 'main'", exc)
        default_branch = "main"

    # Obtener árbol completo del repositorio
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
            "HTTP %s al obtener el árbol de GitHub.\n"
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
            "El árbol de GitHub fue truncado (repositorio muy grande). "
            "Algunos archivos pueden quedar sin descargar."
        )

    blobs = [item for item in tree_data.get("tree", []) if item["type"] == "blob"]
    targets = [
        b for b in blobs
        if Path(b["path"]).suffix.lower() in IMAGE_EXTS | ANNOTATION_EXTS
    ]
    logger.info(
        "Repositorio: %d archivo(s) totales → %d imágenes/anotaciones a descargar",
        len(blobs), len(targets),
    )

    for item in targets:
        rel_path = item["path"]
        raw_url  = (
            f"https://raw.githubusercontent.com/{GITHUB_REPO}"
            f"/{default_branch}/{rel_path}"
        )
        # Preservar la estructura de carpetas del repo
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
            label=label,
        )
        # Pausa suave para no agotar el rate limit
        time.sleep(0.05)

    logger.info("GitHub: completado.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGSHARE
#    URL: https://figshare.com/articles/dataset/Pavement_cracks_from_UAV_imagery-1388/25103138
# ═══════════════════════════════════════════════════════════════════════════════

def download_figshare(registry: Registry, logger: logging.Logger) -> None:
    """
    Descarga desde Figshare usando su API pública v2 (sin autenticación).
    Documentación: https://docs.figshare.com/
    """
    logger.info("═══ [4/4] FIGSHARE ══════════════════════════════════")
    out_dir = SOURCE_DIRS["figshare"]
    out_dir.mkdir(parents=True, exist_ok=True)
    article_url = (
        f"https://figshare.com/articles/dataset/"
        f"Pavement_cracks_from_UAV_imagery-1388/{FIGSHARE_ARTICLE_ID}"
    )

    session = build_session()
    files_api = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"

    logger.info("Obteniendo lista de archivos de Figshare: %s", files_api)
    try:
        resp = session.get(files_api, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        files = resp.json()
    except Exception as exc:
        logger.error("Error al acceder a Figshare API: %s", exc)
        return

    if not isinstance(files, list):
        logger.error("Respuesta inesperada de Figshare: %s", str(files)[:200])
        return

    logger.info("Encontrados %d archivo(s) en Figshare artículo %d", len(files), FIGSHARE_ARTICLE_ID)

    for f in files:
        filename     = f.get("name", f"figshare_{f.get('id', uuid.uuid4().hex)}")
        dl_url       = f.get("download_url", "")
        size_bytes   = f.get("size", 0)

        if not dl_url:
            logger.warning("Sin URL de descarga para: %s", filename)
            continue

        dest = unique_path(out_dir, filename)
        logger.info("Descargando: %s  (%s)", filename, _human_size(size_bytes))
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
                    f"figshare-{FIGSHARE_ARTICLE_ID}", article_url,
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
        )

    logger.info("Figshare: completado.")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger   = setup_logging()
    registry = Registry()

    logger.info(
        "Pavement Crack Dataset Downloader  —  %s",
        datetime.now(timezone.utc).isoformat(),
    )
    logger.info("Directorio de salida: %s", BASE_DIR.resolve())

    for d in SOURCE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    errors: List[tuple] = []

    for name, fn in [
        ("roboflow", download_roboflow),
        ("mendeley",  download_mendeley),
        ("github",    download_github),
        ("figshare",  download_figshare),
    ]:
        try:
            fn(registry, logger)
        except KeyboardInterrupt:
            logger.warning("Descarga interrumpida por el usuario.")
            break
        except Exception as exc:
            logger.exception("Error inesperado en [%s]: %s", name, exc)
            errors.append((name, str(exc)))

    # Guardar registro
    registry.save(logger)

    # Resumen final
    total = len(registry.records)
    logger.info("═" * 60)
    logger.info("RESUMEN FINAL")
    logger.info("  Total de archivos registrados: %d", total)
    for src in SOURCE_DIRS:
        n = sum(1 for r in registry.records if r["source"] == src)
        imgs  = sum(1 for r in registry.records if r["source"] == src and r["file_type"] == "image")
        annos = sum(1 for r in registry.records if r["source"] == src and r["file_type"] == "annotation")
        logger.info("  %-12s %3d archivos  (%d imágenes, %d anotaciones)", src, n, imgs, annos)
    if errors:
        logger.warning("Fuentes con errores: %s", [e[0] for e in errors])
    logger.info("Registro: %s", (BASE_DIR / "download_registry.csv").resolve())
    logger.info("Listo.")


if __name__ == "__main__":
    main()
