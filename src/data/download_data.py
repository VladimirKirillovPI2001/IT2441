"""
Download PlantVillage dataset from Kaggle.
Requires: Kaggle account + API key (kaggle.json in ~/.kaggle/ or %USERPROFILE%\.kaggle\).
Alternative: download manually from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
"""
import os
from pathlib import Path

try:
    import opendatasets as od
except ImportError:
    print("Install: pip install opendatasets")
    raise

from src.config import RAW_DIR

KAGGLE_URL = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
OUTPUT_DIR = RAW_DIR / "PlantVillage"


def download():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Data already at {OUTPUT_DIR}. Skip download.")
        return str(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    od.download(KAGGLE_URL, data_dir=RAW_DIR)
    # opendatasets may create PlantVillage-dataset or similar subdir
    for d in RAW_DIR.iterdir():
        if d.is_dir() and "plant" in d.name.lower():
            return str(d)
    return str(OUTPUT_DIR)


if __name__ == "__main__":
    download()
