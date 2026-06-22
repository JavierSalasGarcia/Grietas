#!/usr/bin/env python3
"""
Fase 2 — Fine-tuning de DINOv2-Base como clasificador Teacher.

Uso:
    pip install torch torchvision transformers scikit-learn matplotlib tqdm
    python train_teacher.py

Salida:
    models/teacher_dinov2_best.pth
    models/teacher_dinov2_config.json
    models/teacher_confusion_matrix.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("dataset_processed")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

STATS_PATH  = DATA_DIR / "dataset_stats.json"

# Hiperparámetros
BATCH_SIZE    = 32
LR_BACKBONE   = 1e-5
LR_HEAD       = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 20
DROPOUT       = 0.3
FROZEN_BLOCKS = 10    # congelar primeros 10 de los 12 bloques del ViT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("models/train_teacher.log")])
log = logging.getLogger("teacher")


# ──────────────────────────────────────────────────────────────────────────────
# MODELO
# ──────────────────────────────────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """
    DINOv2-Base con cabeza de clasificación binaria.
    Los primeros FROZEN_BLOCKS del ViT están congelados.
    """
    def __init__(self, num_classes: int = 2, frozen_blocks: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # Congelar los primeros N bloques del transformer
        modules_to_freeze = list(self.backbone.encoder.layer[:frozen_blocks])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        # Congelar también embeddings y layernorm iniciales
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size   # 768 para dinov2-base
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]   # token [CLS]
        return self.head(cls_token)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET Y DATALOADERS
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(stats: dict) -> tuple:
    """
    Construye DataLoaders para train, val y test.
    El orden de clases lo determina ImageFolder (alfabético):
      0 = Aprobado, 1 = No_Aprobado
    """
    mean = stats["mean_rgb"]
    std  = stats["std_rgb"]

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

    # Calcular pesos de clase para manejar desbalance residual
    class_counts = torch.tensor([
        sum(1 for _, l in train_ds.samples if l == c)
        for c in range(len(train_ds.classes))
    ], dtype=torch.float)
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    log.info("Clases: %s", train_ds.class_to_idx)
    log.info("Pesos de clase: %s", dict(zip(train_ds.classes, class_weights.tolist())))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    return train_dl, val_dl, test_dl, class_weights, train_ds.class_to_idx


# ──────────────────────────────────────────────────────────────────────────────
# LOOP DE ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "preds":    all_preds,
        "labels":   all_labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    stats = json.loads(STATS_PATH.read_text())

    train_dl, val_dl, test_dl, class_weights, class_to_idx = build_dataloaders(stats)

    model     = DINOv2Classifier(frozen_blocks=FROZEN_BLOCKS, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Dos grupos de parámetros con learning rates distintos
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": model.head.parameters(),
         "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_f1   = 0.0
    history   = []

    for epoch in range(1, EPOCHS + 1):
        log.info("Época %d/%d", epoch, EPOCHS)
        train_loss = train_epoch(model, train_dl, criterion, optimizer, scaler, DEVICE)
        val_metrics = evaluate(model, val_dl, criterion, DEVICE)
        scheduler.step()

        report = classification_report(
            val_metrics["labels"], val_metrics["preds"],
            target_names=list(class_to_idx.keys()), output_dict=True
        )
        # Índice 0 = "Aprobado" (alfabético) → recall de grietas
        aprobado_recall = report[list(class_to_idx.keys())[0]]["recall"]
        f1_macro        = report["macro avg"]["f1-score"]

        log.info("  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  "
                 "val_f1=%.3f  aprobado_recall=%.3f",
                 train_loss, val_metrics["loss"], val_metrics["accuracy"],
                 f1_macro, aprobado_recall)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": f1_macro,
            "aprobado_recall": aprobado_recall,
        })

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), MODELS_DIR / "teacher_dinov2_best.pth")
            log.info("  ✓ Mejor modelo guardado (f1=%.4f)", best_f1)

    # Evaluación final en test
    log.info("Evaluando en test set…")
    model.load_state_dict(torch.load(MODELS_DIR / "teacher_dinov2_best.pth"))
    test_metrics = evaluate(model, test_dl, criterion, DEVICE)
    test_report  = classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=list(class_to_idx.keys()), output_dict=True
    )
    log.info("Test accuracy: %.4f", test_metrics["accuracy"])
    log.info("Test report:\n%s",
             classification_report(test_metrics["labels"], test_metrics["preds"],
                                   target_names=list(class_to_idx.keys())))

    # Guardar configuración y métricas
    config = {
        "model": "facebook/dinov2-base",
        "frozen_blocks": FROZEN_BLOCKS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD,
        "batch_size": BATCH_SIZE,
        "class_to_idx": class_to_idx,
        "best_val_f1": best_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_report": test_report,
        "history": history,
    }
    (MODELS_DIR / "teacher_dinov2_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Matriz de confusión
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=list(class_to_idx.keys()))
    disp.plot(cmap="Blues")
    plt.title("Teacher DINOv2 — Confusion Matrix (Test)")
    plt.savefig(MODELS_DIR / "teacher_confusion_matrix.png", dpi=150)
    plt.close()
    log.info("Fase 2 completada. Modelo en: %s", MODELS_DIR / "teacher_dinov2_best.pth")


if __name__ == "__main__":
    main()
