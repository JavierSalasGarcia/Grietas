"""
Tests unitarios para Fase 1 — preprocessing/preprocess_dataset.py

Ejecutar con:
    pytest tests/test_preprocess.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Añadir raíz del proyecto al path para importar el módulo
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.preprocess_dataset import (
    IMAGE_SIZE,
    MAX_IMBALANCE_RATIO,
    RANDOM_SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    balance_classes,
    compute_normalization_stats,
    load_images_from_registry,
    process_split,
    save_splits,
    split_dataset,
    verify_and_filter,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_image_dir(tmp_path: Path) -> Path:
    """Crea imágenes JPEG válidas de prueba en un directorio temporal."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    return img_dir


def _make_image(path: Path, size: tuple = (100, 100), mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size)
    # JPEG no soporta RGBA ni P; usar PNG para modos no-RGB
    fmt = "JPEG" if mode == "RGB" else "PNG"
    # Asegurar extensión coherente con el formato
    save_path = path.with_suffix(".png") if fmt == "PNG" else path
    img.save(save_path, format=fmt)
    return save_path


def _make_registry(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "download_registry.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _sample_df(n_aprobado: int = 6, n_no_aprobado: int = 4,
               tmp_path: Path | None = None) -> pd.DataFrame:
    """Construye un DataFrame de prueba con rutas simuladas."""
    rows = []
    for i in range(n_aprobado):
        rows.append({
            "local_path": f"/fake/path/img_ap_{i}.jpg",
            "file_type": "image",
            "categoria_proyecto": "Aprobado",
            "source": "test",
        })
    for i in range(n_no_aprobado):
        rows.append({
            "local_path": f"/fake/path/img_no_{i}.jpg",
            "file_type": "image",
            "categoria_proyecto": "No Aprobado",
            "source": "test",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Tests de constantes
# ──────────────────────────────────────────────────────────────────────────────

def test_splits_sum_to_one():
    assert abs(SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST - 1.0) < 1e-9


def test_image_size_is_224():
    assert IMAGE_SIZE == (224, 224)


def test_random_seed():
    assert RANDOM_SEED == 42


def test_max_imbalance_ratio():
    assert MAX_IMBALANCE_RATIO == 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: load_images_from_registry
# ──────────────────────────────────────────────────────────────────────────────

def test_load_images_filters_non_image_rows(tmp_path: Path):
    csv_path = _make_registry(tmp_path, [
        {"file_type": "image",       "local_path": "/a.jpg", "categoria_proyecto": "Aprobado",   "source": "s"},
        {"file_type": "annotation",  "local_path": "/b.xml", "categoria_proyecto": "Aprobado",   "source": "s"},
        {"file_type": "image",       "local_path": "/c.jpg", "categoria_proyecto": "No Aprobado","source": "s"},
        {"file_type": "text",        "local_path": "/d.txt", "categoria_proyecto": "Aprobado",   "source": "s"},
    ])
    df = load_images_from_registry(csv_path)
    assert len(df) == 2
    assert set(df["file_type"]) == {"image"}


def test_load_images_filters_unknown_categories(tmp_path: Path):
    csv_path = _make_registry(tmp_path, [
        {"file_type": "image", "local_path": "/a.jpg", "categoria_proyecto": "Aprobado",    "source": "s"},
        {"file_type": "image", "local_path": "/b.jpg", "categoria_proyecto": "Desconocido", "source": "s"},
        {"file_type": "image", "local_path": "/c.jpg", "categoria_proyecto": "No Aprobado", "source": "s"},
    ])
    df = load_images_from_registry(csv_path)
    assert len(df) == 2
    assert "Desconocido" not in df["categoria_proyecto"].values


def test_load_images_drops_null_local_path(tmp_path: Path):
    csv_path = _make_registry(tmp_path, [
        {"file_type": "image", "local_path": None,    "categoria_proyecto": "Aprobado",   "source": "s"},
        {"file_type": "image", "local_path": "/a.jpg","categoria_proyecto": "No Aprobado","source": "s"},
    ])
    df = load_images_from_registry(csv_path)
    assert len(df) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Tests: verify_and_filter
# ──────────────────────────────────────────────────────────────────────────────

def test_verify_keeps_valid_images(tmp_path: Path, tmp_image_dir: Path):
    p1 = _make_image(tmp_image_dir / "good1.jpg")
    p2 = _make_image(tmp_image_dir / "good2.jpg")
    df = pd.DataFrame([
        {"local_path": str(p1), "file_type": "image", "categoria_proyecto": "Aprobado",   "source": "s"},
        {"local_path": str(p2), "file_type": "image", "categoria_proyecto": "No Aprobado","source": "s"},
    ])
    result = verify_and_filter(df)
    assert len(result) == 2


def test_verify_excludes_nonexistent_files(tmp_path: Path):
    df = pd.DataFrame([
        {"local_path": "/nonexistent/path/img.jpg", "file_type": "image",
         "categoria_proyecto": "Aprobado", "source": "s"},
    ])
    result = verify_and_filter(df)
    assert len(result) == 0


def test_verify_excludes_corrupt_images(tmp_path: Path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image at all")
    df = pd.DataFrame([
        {"local_path": str(corrupt), "file_type": "image",
         "categoria_proyecto": "Aprobado", "source": "s"},
    ])
    result = verify_and_filter(df)
    assert len(result) == 0


def test_verify_handles_rgba_images(tmp_path: Path, tmp_image_dir: Path):
    p = _make_image(tmp_image_dir / "rgba.png", mode="RGBA")
    df = pd.DataFrame([
        {"local_path": str(p), "file_type": "image", "categoria_proyecto": "Aprobado", "source": "s"},
    ])
    result = verify_and_filter(df)
    assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Tests: balance_classes
# ──────────────────────────────────────────────────────────────────────────────

def test_balance_no_action_when_within_ratio():
    df = _sample_df(n_aprobado=6, n_no_aprobado=4)
    # ratio = 6/4 = 1.5 ≤ 3.0 → no submuestrear
    result = balance_classes(df, max_ratio=3.0, seed=42)
    assert len(result) == 10


def test_balance_subsamples_majority_class():
    # ratio = 30/5 = 6 > 3 → submuestrear mayoritaria a 5*3 = 15
    df = _sample_df(n_aprobado=30, n_no_aprobado=5)
    result = balance_classes(df, max_ratio=3.0, seed=42)
    counts = result["categoria_proyecto"].value_counts()
    assert counts.max() / counts.min() <= 3.0 + 0.01


def test_balance_preserves_minority_class():
    n_minority = 5
    df = _sample_df(n_aprobado=30, n_no_aprobado=n_minority)
    result = balance_classes(df, max_ratio=3.0, seed=42)
    # "No Aprobado" es la minoría y debe conservarse completa
    no_aprobado_count = (result["categoria_proyecto"] == "No Aprobado").sum()
    assert no_aprobado_count == n_minority


def test_balance_exact_ratio_no_subsampling():
    # ratio exacto = 3.0 → no submuestrear
    df = _sample_df(n_aprobado=15, n_no_aprobado=5)
    result = balance_classes(df, max_ratio=3.0, seed=42)
    assert len(result) == 20


# ──────────────────────────────────────────────────────────────────────────────
# Tests: split_dataset
# ──────────────────────────────────────────────────────────────────────────────

def test_split_sizes_sum_to_total():
    df = _sample_df(n_aprobado=50, n_no_aprobado=50)
    train, val, test = split_dataset(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_train_proportion():
    df = _sample_df(n_aprobado=100, n_no_aprobado=100)
    train, val, test = split_dataset(df, seed=42)
    # Tolerancia ±2%
    assert abs(len(train) / len(df) - SPLIT_TRAIN) < 0.02


def test_split_val_proportion():
    df = _sample_df(n_aprobado=100, n_no_aprobado=100)
    train, val, test = split_dataset(df, seed=42)
    assert abs(len(val) / len(df) - SPLIT_VAL) < 0.02


def test_split_test_proportion():
    df = _sample_df(n_aprobado=100, n_no_aprobado=100)
    train, val, test = split_dataset(df, seed=42)
    assert abs(len(test) / len(df) - SPLIT_TEST) < 0.02


def test_split_stratification():
    """Ambas clases deben aparecer en cada split."""
    df = _sample_df(n_aprobado=50, n_no_aprobado=50)
    train, val, test = split_dataset(df, seed=42)
    for split in (train, val, test):
        cats = set(split["categoria_proyecto"])
        assert "Aprobado" in cats
        assert "No Aprobado" in cats


def test_split_no_overlap():
    """Ninguna imagen puede aparecer en más de un split."""
    df = _sample_df(n_aprobado=50, n_no_aprobado=50)
    train, val, test = split_dataset(df, seed=42)
    paths_train = set(train["local_path"])
    paths_val   = set(val["local_path"])
    paths_test  = set(test["local_path"])
    assert paths_train.isdisjoint(paths_val)
    assert paths_train.isdisjoint(paths_test)
    assert paths_val.isdisjoint(paths_test)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: process_split
# ──────────────────────────────────────────────────────────────────────────────

def test_process_split_creates_resized_images(tmp_path: Path, tmp_image_dir: Path,
                                              monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_path = _make_image(tmp_image_dir / "sample.jpg", size=(640, 480))
    df = pd.DataFrame([{
        "local_path": str(img_path),
        "file_type": "image",
        "categoria_proyecto": "Aprobado",
        "source": "test",
    }])

    import preprocessing.preprocess_dataset as mod
    original_output = mod.OUTPUT_DIR
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"

    result = process_split(df, "train")

    mod.OUTPUT_DIR = original_output

    assert len(result) == 1
    out_path = Path(result.iloc[0]["processed_path"])
    assert out_path.exists()
    with Image.open(out_path) as img:
        assert img.size == IMAGE_SIZE


def test_process_split_converts_rgba_to_rgb(tmp_path: Path, tmp_image_dir: Path,
                                            monkeypatch):
    monkeypatch.chdir(tmp_path)

    img_path = _make_image(tmp_image_dir / "rgba_img.png", mode="RGBA")
    df = pd.DataFrame([{
        "local_path": str(img_path),
        "file_type": "image",
        "categoria_proyecto": "No Aprobado",
        "source": "test",
    }])

    import preprocessing.preprocess_dataset as mod
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"

    result = process_split(df, "val")
    assert len(result) == 1
    out_path = Path(result.iloc[0]["processed_path"])
    with Image.open(out_path) as img:
        assert img.mode == "RGB"


def test_process_split_no_aprobado_folder_name(tmp_path: Path, tmp_image_dir: Path,
                                               monkeypatch):
    """La carpeta debe llamarse 'No_Aprobado' (con guion bajo, no espacio)."""
    monkeypatch.chdir(tmp_path)

    img_path = _make_image(tmp_image_dir / "img.jpg")
    df = pd.DataFrame([{
        "local_path": str(img_path),
        "file_type": "image",
        "categoria_proyecto": "No Aprobado",
        "source": "test",
    }])

    import preprocessing.preprocess_dataset as mod
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"

    result = process_split(df, "train")
    out_path = Path(result.iloc[0]["processed_path"])
    assert "No_Aprobado" in str(out_path)
    assert "No Aprobado" not in str(out_path)


def test_process_split_handles_name_collisions(tmp_path: Path, tmp_image_dir: Path,
                                               monkeypatch):
    """Si dos fuentes tienen el mismo nombre de archivo, ambas deben guardarse."""
    monkeypatch.chdir(tmp_path)

    p1 = _make_image(tmp_image_dir / "dup1" / "photo.jpg")
    p2 = _make_image(tmp_image_dir / "dup2" / "photo.jpg")

    df = pd.DataFrame([
        {"local_path": str(p1), "file_type": "image", "categoria_proyecto": "Aprobado", "source": "s1"},
        {"local_path": str(p2), "file_type": "image", "categoria_proyecto": "Aprobado", "source": "s2"},
    ])

    import preprocessing.preprocess_dataset as mod
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"

    result = process_split(df, "train")
    assert len(result) == 2
    # Ambos archivos deben existir y ser distintos
    paths = [Path(r["processed_path"]) for _, r in result.iterrows()]
    assert paths[0] != paths[1]
    assert all(p.exists() for p in paths)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: compute_normalization_stats
# ──────────────────────────────────────────────────────────────────────────────

def test_normalization_stats_shape(tmp_path: Path, tmp_image_dir: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    paths = []
    for i in range(5):
        p = _make_image(tmp_image_dir / f"img_{i}.jpg")
        paths.append(str(p))

    df = pd.DataFrame({"processed_path": paths})
    stats = compute_normalization_stats(df)

    assert "mean_rgb" in stats
    assert "std_rgb" in stats
    assert len(stats["mean_rgb"]) == 3
    assert len(stats["std_rgb"]) == 3


def test_normalization_stats_range(tmp_path: Path, tmp_image_dir: Path, monkeypatch):
    """Media y std deben estar en [0, 1] para imágenes normalizadas."""
    monkeypatch.chdir(tmp_path)

    for i in range(5):
        _make_image(tmp_image_dir / f"img_{i}.jpg")

    df = pd.DataFrame({"processed_path": [str(p) for p in tmp_image_dir.glob("*.jpg")]})
    stats = compute_normalization_stats(df)

    for v in stats["mean_rgb"]:
        assert 0.0 <= v <= 1.0
    for v in stats["std_rgb"]:
        assert 0.0 <= v <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: save_splits
# ──────────────────────────────────────────────────────────────────────────────

def _minimal_df(n: int, cat: str = "Aprobado") -> pd.DataFrame:
    return pd.DataFrame({
        "processed_path":    [f"/out/img_{i}.jpg" for i in range(n)],
        "categoria_proyecto": [cat] * n,
        "source":            ["test"] * n,
        "local_path":        [f"/src/img_{i}.jpg" for i in range(n)],
    })


def test_save_splits_creates_csv_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import preprocessing.preprocess_dataset as mod
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"
    mod.OUTPUT_DIR.mkdir()

    df_train = _minimal_df(70)
    df_val   = _minimal_df(15)
    df_test  = _minimal_df(15)
    stats    = {"mean_rgb": [0.5, 0.5, 0.5], "std_rgb": [0.2, 0.2, 0.2]}

    save_splits(df_train, df_val, df_test, stats)

    assert (mod.OUTPUT_DIR / "split_train.csv").exists()
    assert (mod.OUTPUT_DIR / "split_val.csv").exists()
    assert (mod.OUTPUT_DIR / "split_test.csv").exists()
    assert (mod.OUTPUT_DIR / "dataset_stats.json").exists()


def test_save_splits_stats_json_content(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import preprocessing.preprocess_dataset as mod
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"
    mod.OUTPUT_DIR.mkdir()

    df_train = _minimal_df(70)
    df_val   = _minimal_df(15)
    df_test  = _minimal_df(15)
    stats    = {"mean_rgb": [0.45, 0.40, 0.35], "std_rgb": [0.22, 0.21, 0.20]}

    save_splits(df_train, df_val, df_test, stats)

    data = json.loads((mod.OUTPUT_DIR / "dataset_stats.json").read_text())
    assert data["n_train"] == 70
    assert data["n_val"] == 15
    assert data["n_test"] == 15
    assert data["image_size"] == [224, 224]
    assert data["random_seed"] == RANDOM_SEED
    assert "mean_rgb" in data
    assert "std_rgb" in data


def test_save_splits_csv_columns(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import preprocessing.preprocess_dataset as mod
    mod.OUTPUT_DIR = tmp_path / "dataset_processed"
    mod.OUTPUT_DIR.mkdir()

    df_train = _minimal_df(10)
    df_val   = _minimal_df(3)
    df_test  = _minimal_df(3)
    stats    = {"mean_rgb": [0.5, 0.5, 0.5], "std_rgb": [0.2, 0.2, 0.2]}

    save_splits(df_train, df_val, df_test, stats)

    expected_cols = {"processed_path", "categoria_proyecto", "source", "local_path"}
    for split_name in ("train", "val", "test"):
        df = pd.read_csv(mod.OUTPUT_DIR / f"split_{split_name}.csv")
        assert expected_cols.issubset(set(df.columns))
