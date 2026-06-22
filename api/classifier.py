"""
Clasificador binario usando el modelo ONNX del Student (Fase 3).
Se carga una sola vez al iniciar el servidor (singleton).
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from api.config import DATASET_STATS, ONNX_MODEL_PATH, STUDENT_CFG_PATH


class CrackClassifier:
    def __init__(self):
        self._session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        cfg = json.loads(STUDENT_CFG_PATH.read_text())
        stats = json.loads(Path(DATASET_STATS).read_text())

        self._idx_to_class = {v: k for k, v in cfg["class_to_idx"].items()}
        self._mean = np.array(stats["mean_rgb"], dtype=np.float32)
        self._std = np.array(stats["std_rgb"], dtype=np.float32)

    def predict(self, image_path: str) -> dict:
        """
        Clasifica una imagen.
        Retorna {"categoria": str, "confianza": float, "probabilidades": dict}
        """
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((224, 224))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)

        logits = self._session.run(None, {"image": arr})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))

        return {
            "categoria": self._idx_to_class[idx],
            "confianza": float(probs[idx]),
            "probabilidades": {
                self._idx_to_class[i]: float(p) for i, p in enumerate(probs)
            },
        }


def _load_classifier() -> CrackClassifier | None:
    """Carga el clasificador si el modelo ONNX existe, si no retorna None."""
    if not ONNX_MODEL_PATH.exists() or not STUDENT_CFG_PATH.exists():
        return None
    try:
        return CrackClassifier()
    except Exception:
        return None


classifier: CrackClassifier | None = _load_classifier()
