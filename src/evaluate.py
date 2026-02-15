"""Evaluation: accuracy, F1, confusion matrix. Run: python -m src.evaluate"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import DEVICE, MODEL_DIR, RESULTS_DIR, REPORTS_DIR
from src.data.dataset import get_dataloaders
from src.model import build_model


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    ckpt = torch.load(MODEL_DIR / "best.pt", map_location=device)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("Accuracy:", round(accuracy, 4))
    print("F1 (macro):", round(f1_macro, 4))
    print("F1 (weighted):", round(f1_weighted, 4))
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png to results/")

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.4f}\n")
    print("Saved metrics.txt to results/")


if __name__ == "__main__":
    main()
