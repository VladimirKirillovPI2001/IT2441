"""Configuration for plant classification project."""
import sys
from pathlib import Path

# Paths: при запуске из exe (PyInstaller) — папка с exe
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create dirs
for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Data
# PlantVillage: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# After download, expected structure: data/raw/PlantVillage/ (train/val or by class folders)
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 for Windows to avoid multiprocessing issues; set 4+ on Linux
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Training
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda"  # or "cpu"

# Model
MODEL_NAME = "efficientnet_b0"  # or "resnet18", "efficientnet_b0"
