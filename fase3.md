# Fase 3 — Destilación del Modelo Student (MobileNetV3) `[PENDIENTE]`

## Contexto del proyecto

Sistema de clasificación binaria de grietas en calles de Toluca. Esta fase
aplica **Knowledge Distillation**: el modelo Teacher (DINOv2-Base, Fase 2)
transfiere su conocimiento a un modelo Student ligero (MobileNetV3-Small, 2.5 M
parámetros) que se desplegará en el backend de la API para inferencia en tiempo real.

---

## Prerrequisitos

- Fase 1 completada: `dataset_processed/` con splits y `dataset_stats.json`
- Fase 2 completada: `models/teacher_dinov2_best.pth` y `teacher_dinov2_config.json`
- `pip install torch torchvision scikit-learn matplotlib tqdm`

---

## Objetivo

Entrenar MobileNetV3-Small como Student usando la función de pérdida de
destilación de Hinton (2015): combinación de CrossEntropy sobre las etiquetas
duras y KL Divergence sobre las distribuciones de probabilidad suavizadas del Teacher.

**Criterio de aceptación**:
- `val_accuracy ≥ 90 %`
- Tamaño del archivo `.onnx` ≤ 15 MB
- Tiempo de inferencia ≤ 200 ms en CPU para una imagen 224×224

---

## Principio de la destilación

```
Teacher (DINOv2, congelado)        Student (MobileNetV3, en entrenamiento)
         │                                   │
    imagen_batch ──────────────────────────►│
         │                                   │
    logits_teacher                      logits_student
         │                                   │
         ├── softmax(/ T) ──► KL_Div ◄── log_softmax(/ T)
         │                        │
         │                     loss_suave  (T² para escalar gradientes)
         │                        │
         └────────────────────────┤
                                  │
                          CrossEntropy(logits_student, hard_labels)
                                  │
                             loss_dura
                                  │
                    loss_total = α * loss_dura + (1-α) * loss_suave
```

**Parámetros fijados**:
- `T = 3` (temperatura): suaviza la distribución del Teacher para que el Student
  aprenda las "similitudes entre clases" además de la clase correcta
- `α = 0.5`: peso igual entre pérdida dura y pérdida suave

---

## Archivo a crear

```
train_student.py
```

---

## Implementación completa

```python
#!/usr/bin/env python3
"""
Fase 3 — Knowledge Distillation de DINOv2 (Teacher) a MobileNetV3-Small (Student).

Uso:
    python train_student.py

Prerequisitos:
    models/teacher_dinov2_best.pth
    models/teacher_dinov2_config.json
    dataset_processed/  (split_train.csv, split_val.csv, split_test.csv)

Salida:
    models/student_mobilenetv3_best.pth
    models/student_mobilenetv3.onnx
    models/student_config.json
    models/student_confusion_matrix.png
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from train_teacher import DINOv2Classifier   # reutilizar definición del Teacher

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("dataset_processed")
MODELS_DIR  = Path("models")

# Hiperparámetros de destilación — NO cambiar sin documentar el motivo
T           = 3       # temperatura de destilación
ALPHA       = 0.5     # peso de la pérdida dura vs. suave
LR          = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS      = 30      # más épocas que el Teacher porque el Student es más simple
BATCH_SIZE  = 64      # MobileNetV3 es ligero, caben batches más grandes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("models/train_student.log")])
log = logging.getLogger("student")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PÉRDIDA DE DESTILACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      hard_labels:    torch.Tensor,
                      T: float = 3.0,
                      alpha: float = 0.5) -> torch.Tensor:
    """
    Pérdida de destilación de Hinton et al. (2015).

    student_logits: (batch, num_classes) — salida del Student (sin softmax)
    teacher_logits: (batch, num_classes) — salida del Teacher (sin softmax)
    hard_labels:    (batch,) — etiquetas ground-truth (enteros)
    T:              temperatura (suaviza distribuciones)
    alpha:          peso de la pérdida dura (0=solo destilación, 1=solo CE)
    """
    # Pérdida suave: KL entre distribuciones suavizadas por temperatura
    soft_teacher = F.softmax(teacher_logits / T, dim=1).detach()
    log_soft_student = F.log_softmax(student_logits / T, dim=1)
    # Multiplicar por T² para escalar los gradientes correctamente
    loss_soft = F.kl_div(log_soft_student, soft_teacher,
                         reduction="batchmean") * (T ** 2)

    # Pérdida dura: CrossEntropy sobre las etiquetas reales
    loss_hard = F.cross_entropy(student_logits, hard_labels)

    return alpha * loss_hard + (1.0 - alpha) * loss_soft


# ──────────────────────────────────────────────────────────────────────────────
# MODELO STUDENT
# ──────────────────────────────────────────────────────────────────────────────

def build_student(num_classes: int = 2) -> nn.Module:
    """
    MobileNetV3-Small con cabeza de clasificación para num_classes.
    Preentrenado en ImageNet → solo reemplazar la capa final.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # La cabeza original termina en 1000 (ImageNet)
    # classifier[-1] es un Linear(1024 → 1000)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADERS
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(stats: dict) -> tuple:
    mean = stats["mean_rgb"]
    std  = stats["std_rgb"]

    # Augmentation en entrenamiento — idéntica a Fase 2 para consistencia
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    return train_dl, val_dl, test_dl, train_ds.class_to_idx


# ──────────────────────────────────────────────────────────────────────────────
# LOOPS DE ENTRENAMIENTO Y EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(student, teacher, loader, optimizer, scaler, device) -> float:
    student.train()
    teacher.eval()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=(device == "cuda")):
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = distillation_loss(student_logits, teacher_logits, labels,
                                     T=T, alpha=ALPHA)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "preds":    all_preds,
        "labels":   all_labels,
        "probs":    all_probs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTAR A ONNX
# ──────────────────────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, onnx_path: Path):
    """
    Exporta el Student a formato ONNX para despliegue en el backend FastAPI.
    El modelo en ONNX acepta lotes de imágenes 224×224 normalizadas.
    """
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"},
                      "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    log.info("ONNX exportado: %s (%.1f MB)", onnx_path, size_mb)
    assert size_mb <= 15.0, f"Modelo ONNX {size_mb:.1f} MB > límite 15 MB"


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK DE LATENCIA
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_latency(onnx_path: Path, n_runs: int = 100) -> float:
    """
    Mide la latencia media de inferencia en CPU con onnxruntime.
    Instalar: pip install onnxruntime
    Retorna: latencia media en milisegundos.
    """
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        # Calentamiento
        for _ in range(10):
            sess.run(None, {"image": dummy})
        # Medición
        start = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {"image": dummy})
        elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
        log.info("Latencia media (CPU): %.1f ms por imagen", elapsed_ms)
        assert elapsed_ms <= 200, f"Latencia {elapsed_ms:.0f} ms > 200 ms"
        return elapsed_ms
    except ImportError:
        log.warning("onnxruntime no instalado — saltar benchmark de latencia")
        return -1.0


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    stats = json.loads((DATA_DIR / "dataset_stats.json").read_text())
    teacher_cfg = json.loads((MODELS_DIR / "teacher_dinov2_config.json").read_text())

    train_dl, val_dl, test_dl, class_to_idx = build_dataloaders(stats)

    # Cargar Teacher congelado
    teacher = DINOv2Classifier(
        frozen_blocks=teacher_cfg["frozen_blocks"],
        dropout=teacher_cfg["dropout"],
    ).to(DEVICE)
    teacher.load_state_dict(torch.load(MODELS_DIR / "teacher_dinov2_best.pth",
                                       map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log.info("Teacher cargado y congelado.")

    # Inicializar Student
    student   = build_student(num_classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_acc = 0.0
    history  = []

    for epoch in range(1, EPOCHS + 1):
        log.info("Época %d/%d", epoch, EPOCHS)
        train_loss = train_epoch(student, teacher, train_dl,
                                 optimizer, scaler, DEVICE)
        val_m = evaluate(student, val_dl, DEVICE)
        scheduler.step()

        report = classification_report(
            val_m["labels"], val_m["preds"],
            target_names=list(class_to_idx.keys()), output_dict=True
        )
        f1_macro = report["macro avg"]["f1-score"]
        log.info("  train_loss=%.4f  val_acc=%.3f  val_f1=%.3f",
                 train_loss, val_m["accuracy"], f1_macro)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_accuracy": val_m["accuracy"], "val_f1": f1_macro})

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            torch.save(student.state_dict(),
                       MODELS_DIR / "student_mobilenetv3_best.pth")
            log.info("  ✓ Mejor Student guardado (acc=%.4f)", best_acc)

    # Evaluación final en test
    student.load_state_dict(
        torch.load(MODELS_DIR / "student_mobilenetv3_best.pth", map_location=DEVICE)
    )
    test_m = evaluate(student, test_dl, DEVICE)
    test_report = classification_report(
        test_m["labels"], test_m["preds"],
        target_names=list(class_to_idx.keys()), output_dict=True
    )
    log.info("Test accuracy: %.4f", test_m["accuracy"])
    log.info("Test report:\n%s",
             classification_report(test_m["labels"], test_m["preds"],
                                   target_names=list(class_to_idx.keys())))

    # Exportar a ONNX
    onnx_path = MODELS_DIR / "student_mobilenetv3.onnx"
    export_onnx(student.cpu(), onnx_path)
    latency_ms = benchmark_latency(onnx_path)

    # Guardar configuración
    config = {
        "model": "mobilenet_v3_small",
        "distillation_T": T,
        "distillation_alpha": ALPHA,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "class_to_idx": class_to_idx,
        "best_val_accuracy": best_acc,
        "test_accuracy": test_m["accuracy"],
        "test_report": test_report,
        "onnx_size_mb": onnx_path.stat().st_size / (1024 ** 2),
        "latency_cpu_ms": latency_ms,
        "history": history,
    }
    (MODELS_DIR / "student_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Matriz de confusión
    cm = confusion_matrix(test_m["labels"], test_m["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=list(class_to_idx.keys()))
    disp.plot(cmap="Greens")
    plt.title("Student MobileNetV3 — Confusion Matrix (Test)")
    plt.savefig(MODELS_DIR / "student_confusion_matrix.png", dpi=150)
    plt.close()

    log.info("Fase 3 completada. Modelo ONNX: %s", onnx_path)


if __name__ == "__main__":
    main()
```

---

## Estructura de salida

```
models/
  student_mobilenetv3_best.pth     ← pesos PyTorch (para futuro fine-tuning)
  student_mobilenetv3.onnx         ← modelo de producción (≤ 15 MB)
  student_config.json              ← métricas, hiperparámetros, latencia
  student_confusion_matrix.png
  train_student.log
```

---

## Verificación de la fase

```python
import json, numpy as np
from pathlib import Path

cfg = json.loads(Path("models/student_config.json").read_text())
print(f"Test accuracy:   {cfg['test_accuracy']:.4f}")
print(f"ONNX size:       {cfg['onnx_size_mb']:.1f} MB")
print(f"CPU latency:     {cfg['latency_cpu_ms']:.1f} ms")

assert cfg["test_accuracy"]  >= 0.90,  "Accuracy insuficiente"
assert cfg["onnx_size_mb"]   <= 15.0,  "Modelo demasiado grande"
assert cfg["latency_cpu_ms"] <= 200.0, "Demasiado lento para producción"

# Probar inferencia con onnxruntime
import onnxruntime as ort
sess  = ort.InferenceSession("models/student_mobilenetv3.onnx")
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
out   = sess.run(None, {"image": dummy})
print("Logits shape:", out[0].shape)   # esperado: (1, 2)
print("Fase 3 verificada ✓")
```

---

## Cómo usar el modelo en producción (Fase 5)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Cargar sesión (una vez al arrancar el servidor)
SESSION = ort.InferenceSession("models/student_mobilenetv3.onnx",
                                providers=["CPUExecutionProvider"])
CLASS_NAMES = ["Aprobado", "No_Aprobado"]   # orden alfabético = ImageFolder

# Inferencia
def classify_image(pil_image: Image.Image,
                   mean: list, std: list) -> dict:
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis]    # (1, 3, 224, 224)
    logits  = SESSION.run(None, {"image": arr})[0][0]
    probs   = np.exp(logits) / np.exp(logits).sum()
    pred_idx = int(np.argmax(probs))
    return {
        "categoria":  CLASS_NAMES[pred_idx],
        "confianza":  float(probs[pred_idx]),
        "probabilidades": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
    }
```

---

## Siguiente paso

**Fase 4** — PWA Frontend (`fase4.md`): captura de imagen y metadatos en el navegador.
**Fase 5** — Backend API (`fase5.md`): integra el modelo ONNX en FastAPI.
Estas dos fases pueden desarrollarse en paralelo con la Fase 3.
