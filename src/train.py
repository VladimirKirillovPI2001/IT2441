"""Training script. Run from project root: python -m src.train"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import EPOCHS, LR, DEVICE, MODEL_DIR, REPORTS_DIR
from src.data.dataset import get_dataloaders
from src.model import build_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / n)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print("Classes:", num_classes)

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"Epoch {epoch}: train loss={tl:.4f} acc={ta:.4f} | val loss={vl:.4f} acc={va:.4f}")

        if va > best_acc:
            best_acc = va
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va,
                "class_names": class_names,
            }, MODEL_DIR / "best.pt")
            print("  -> best model saved")

    torch.save(history, REPORTS_DIR / "history.pt")
    print("Done. Best val accuracy:", best_acc)


if __name__ == "__main__":
    main()
