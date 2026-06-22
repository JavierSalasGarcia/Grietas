import os
from pathlib import Path

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://grietas:grietas@localhost:5432/grietas_db"
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
ONNX_MODEL_PATH = MODELS_DIR / "student_mobilenetv3.onnx"
STUDENT_CFG_PATH = MODELS_DIR / "student_config.json"
DATASET_STATS = Path("dataset_processed/dataset_stats.json")

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "uploads"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.5"))
