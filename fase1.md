# Fase 1 — Preprocesamiento y Validación del Dataset `[PENDIENTE]`

## Contexto del proyecto

Sistema de clasificación binaria de grietas en calles de Toluca. Un modelo de IA
decide si cada fotografía ciudadana muestra una grieta real (**Aprobado**) o no
(**No Aprobado**). Esta fase convierte el dataset crudo descargado en Fase 0 en
un dataset limpio, balanceado y dividido listo para entrenamiento.

---

## Prerrequisitos

- Fase 0 completada: existe `pavement_crack_datasets/download_registry.csv`
- Existen imágenes en `pavement_crack_datasets/0{1..5}_*/`
- `pip install pandas Pillow tqdm numpy scikit-learn`

---

## Objetivo

1. Detectar y descartar imágenes corruptas
2. Filtrar solo archivos de tipo imagen (ignorar anotaciones y archivos de texto)
3. Redimensionar a 224×224 px (requerido por DINOv2 y MobileNetV3 en Fase 2 y 3)
4. Balancear clases si el desbalance supera 3:1
5. Dividir en train / val / test (70 / 15 / 15)
6. Calcular estadísticas de normalización (media y desviación estándar RGB)

---

## Archivo a crear

```
preprocess_dataset.py
```

---

## Implementación completa

```python
#!/usr/bin/env python3
"""
Fase 1 — Preprocesamiento y validación del dataset.

Lee download_registry.csv, verifica integridad de imágenes,
redimensiona a 224x224, balancea y divide en train/val/test.

Uso:
    pip install pandas Pillow tqdm numpy scikit-learn
    python preprocess_dataset.py
"""

import json
import shutil
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

REGISTRY_CSV   = Path("pavement_crack_datasets/download_registry.csv")
OUTPUT_DIR     = Path("dataset_processed")
IMAGE_SIZE     = (224, 224)    # tamaño requerido por DINOv2 y MobileNetV3

# Splits: deben sumar 1.0
SPLIT_TRAIN    = 0.70
SPLIT_VAL      = 0.15
SPLIT_TEST     = 0.15

RANDOM_SEED    = 42

# Si la clase mayoritaria supera este ratio sobre la minoritaria → submuestrear
MAX_IMBALANCE_RATIO = 3.0

CORRUPT_LOG    = Path("pavement_crack_datasets/corrupt_files.txt")


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pavement_crack_datasets/preprocess.log"),
    ],
)
log = logging.getLogger("preprocess")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 — Cargar registro y filtrar imágenes
# ──────────────────────────────────────────────────────────────────────────────

def load_images_from_registry(csv_path: Path) -> pd.DataFrame:
    """Lee el registro y devuelve solo las filas de tipo imagen."""
    df = pd.read_csv(csv_path)
    images = df[df["file_type"] == "image"].copy()
    images = images[images["local_path"].notna()]
    images = images[images["categoria_proyecto"].isin(["Aprobado", "No Aprobado"])]
    log.info("Registro total: %d filas → %d imágenes válidas", len(df), len(images))
    return images.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 — Verificar integridad y filtrar corruptas
# ──────────────────────────────────────────────────────────────────────────────

def verify_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta abrir cada imagen con PIL.
    Las corruptas se registran en corrupt_files.txt y se excluyen.
    """
    valid_indices = []
    corrupt_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verificando imágenes"):
        path = Path(row["local_path"])
        if not path.exists():
            corrupt_paths.append(f"NO_EXISTE: {path}")
            continue
        try:
            with Image.open(path) as img:
                img.verify()          # detecta truncado / formato inválido
            # verify() consume el objeto; reabrir para confirmar que se puede leer
            with Image.open(path) as img:
                img.load()
            valid_indices.append(idx)
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            corrupt_paths.append(f"CORRUPTA: {path}  ({e})")

    if corrupt_paths:
        CORRUPT_LOG.write_text("\n".join(corrupt_paths), encoding="utf-8")
        log.warning("%d imágenes corruptas o faltantes → ver %s",
                    len(corrupt_paths), CORRUPT_LOG)

    clean = df.loc[valid_indices].reset_index(drop=True)
    log.info("Imágenes válidas tras verificación: %d", len(clean))
    return clean


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 — Balancear clases
# ──────────────────────────────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, max_ratio: float, seed: int) -> pd.DataFrame:
    """
    Si la clase mayoritaria supera max_ratio veces la minoritaria,
    submuestrea la mayoritaria hasta ese límite.
    Nunca elimina muestras de la clase minoritaria.
    """
    counts = df["categoria_proyecto"].value_counts()
    log.info("Distribución antes de balancear: %s", counts.to_dict())

    n_min = counts.min()
    n_max = counts.max()

    if n_max / n_min > max_ratio:
        target = int(n_min * max_ratio)
        log.info("Desbalance %.1fx > límite %.1fx → submuestreando a %d por clase mayor",
                 n_max / n_min, max_ratio, target)
        clase_mayor = counts.idxmax()
        mayoritaria = df[df["categoria_proyecto"] == clase_mayor].sample(
            target, random_state=seed
        )
        minoritaria = df[df["categoria_proyecto"] != clase_mayor]
        df = pd.concat([mayoritaria, minoritaria]).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)
    else:
        log.info("Desbalance %.1fx ≤ límite %.1fx → no se submuestrea",
                 n_max / n_min, max_ratio)

    log.info("Distribución final: %s",
             df["categoria_proyecto"].value_counts().to_dict())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 4 — Dividir en splits
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(df: pd.DataFrame, seed: int) -> tuple:
    """
    Divide en train/val/test respetando la proporción de clases (stratify).
    Retorna (df_train, df_val, df_test).
    """
    # Primer corte: train vs. (val+test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(SPLIT_VAL + SPLIT_TEST),
        stratify=df["categoria_proyecto"],
        random_state=seed,
    )
    # Segundo corte: val vs. test (proporción relativa dentro del 30% restante)
    relative_test = SPLIT_TEST / (SPLIT_VAL + SPLIT_TEST)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test,
        stratify=df_temp["categoria_proyecto"],
        random_state=seed,
    )
    log.info("Splits → train: %d  |  val: %d  |  test: %d",
             len(df_train), len(df_val), len(df_test))
    return df_train, df_val, df_test


# ──────────────────────────────────────────────────────────────────────────────
# PASO 5 — Copiar y redimensionar imágenes
# ──────────────────────────────────────────────────────────────────────────────

def process_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Redimensiona cada imagen a IMAGE_SIZE y la copia en:
      dataset_processed/{split_name}/{categoria}/
    La categoría 'No Aprobado' se convierte a 'No_Aprobado' (sin espacio).
    Devuelve df con columna 'processed_path' añadida.
    """
    processed_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Procesando {split_name}"):
        src   = Path(row["local_path"])
        cat   = row["categoria_proyecto"].replace(" ", "_")   # "No_Aprobado"
        dest_dir = OUTPUT_DIR / split_name / cat
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / src.name
        # Garantizar nombre único dentro del split
        n = 1
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1

        try:
            with Image.open(src) as img:
                # Convertir a RGB (elimina canal alpha, maneja paletas, etc.)
                img_rgb = img.convert("RGB")
                img_rgb = img_rgb.resize(IMAGE_SIZE, Image.LANCZOS)
                img_rgb.save(dest, format="JPEG", quality=92)
            processed_paths.append(str(dest))
        except Exception as e:
            log.warning("Error procesando %s: %s", src.name, e)
            processed_paths.append("")

    df = df.copy()
    df["processed_path"] = processed_paths
    return df[df["processed_path"] != ""].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# PASO 6 — Calcular estadísticas de normalización
# ──────────────────────────────────────────────────────────────────────────────

def compute_normalization_stats(df_train: pd.DataFrame) -> dict:
    """
    Calcula media y desviación estándar RGB del conjunto de entrenamiento.
    Necesario para normalizar las imágenes durante el entrenamiento (Fase 2).
    Usa muestreo de hasta 2 000 imágenes para eficiencia.
    """
    log.info("Calculando estadísticas de normalización (muestra de entrenamiento)…")
    sample = df_train.sample(min(2000, len(df_train)), random_state=RANDOM_SEED)
    pixels = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Stats RGB"):
        try:
            with Image.open(row["processed_path"]) as img:
                arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                pixels.append(arr.reshape(-1, 3))
        except Exception:
            continue

    all_pixels = np.concatenate(pixels, axis=0)
    mean = all_pixels.mean(axis=0).tolist()
    std  = all_pixels.std(axis=0).tolist()
    log.info("RGB mean: %s  |  RGB std: %s", mean, std)
    return {"mean_rgb": mean, "std_rgb": std}


# ──────────────────────────────────────────────────────────────────────────────
# PASO 7 — Guardar splits y estadísticas
# ──────────────────────────────────────────────────────────────────────────────

def save_splits(df_train, df_val, df_test, stats: dict):
    """Guarda los splits en CSV y las estadísticas en JSON."""
    cols = ["processed_path", "categoria_proyecto", "source", "local_path"]

    df_train[cols].to_csv(OUTPUT_DIR / "split_train.csv", index=False)
    df_val[cols].to_csv(OUTPUT_DIR / "split_val.csv",   index=False)
    df_test[cols].to_csv(OUTPUT_DIR / "split_test.csv",  index=False)

    stats.update({
        "n_train": len(df_train),
        "n_val":   len(df_val),
        "n_test":  len(df_test),
        "n_train_aprobado":    int((df_train["categoria_proyecto"] == "Aprobado").sum()),
        "n_train_no_aprobado": int((df_train["categoria_proyecto"] == "No Aprobado").sum()),
        "image_size":   list(IMAGE_SIZE),
        "random_seed":  RANDOM_SEED,
    })
    (OUTPUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    log.info("Splits y estadísticas guardados en %s/", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df        = load_images_from_registry(REGISTRY_CSV)
    df        = verify_and_filter(df)
    df        = balance_classes(df, MAX_IMBALANCE_RATIO, RANDOM_SEED)
    df_train, df_val, df_test = split_dataset(df, RANDOM_SEED)

    df_train = process_split(df_train, "train")
    df_val   = process_split(df_val,   "val")
    df_test  = process_split(df_test,  "test")

    stats = compute_normalization_stats(df_train)
    save_splits(df_train, df_val, df_test, stats)

    log.info("Fase 1 completada. Dataset en: %s/", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
```

---

## Estructura de salida

```
dataset_processed/
  train/
    Aprobado/         ← imágenes 224×224 JPEG
    No_Aprobado/
  val/
    Aprobado/
    No_Aprobado/
  test/
    Aprobado/
    No_Aprobado/
  split_train.csv     ← columnas: processed_path, categoria_proyecto, source, local_path
  split_val.csv
  split_test.csv
  dataset_stats.json  ← {mean_rgb, std_rgb, n_train, n_val, n_test, image_size, ...}
```

---

## Verificación de la fase

```python
import json, pandas as pd
from pathlib import Path

# Verificar splits
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"dataset_processed/split_{split}.csv")
    cats = df["categoria_proyecto"].value_counts()
    print(f"{split}: {len(df)} imágenes → {cats.to_dict()}")
    # Verificar que los archivos existen
    missing = [p for p in df["processed_path"] if not Path(p).exists()]
    assert not missing, f"{len(missing)} archivos faltantes en {split}"

# Verificar estadísticas
stats = json.loads(Path("dataset_processed/dataset_stats.json").read_text())
assert len(stats["mean_rgb"]) == 3
assert stats["n_train"] > 0
print("Stats:", stats)
```

---

## Notas importantes

- **No borrar el directorio de origen** `pavement_crack_datasets/` al terminar:
  los `local_path` en los CSVs de split apuntan a esos archivos originales.
  La columna `processed_path` apunta a las copias redimensionadas.
- La conversión a RGB con `img.convert("RGB")` maneja correctamente imágenes
  en escala de grises (las convierte a 3 canales) y con canal alpha (RGBA → RGB).
- Si se agrega más datos en el futuro, re-ejecutar esta fase completa.
  Los splits anteriores se sobreescriben.

---

## Siguiente paso

**Fase 2** — Entrenamiento del modelo Teacher (`fase2.md`).
Lee `dataset_processed/` y `dataset_stats.json` para hacer fine-tuning de DINOv2.
