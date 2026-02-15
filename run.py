"""
Точка входа для exe. Запуск: run.py train | evaluate
При запуске из exe: PROJECT_ROOT = папка с exe (рядом data/, models/).
"""
import sys
from pathlib import Path

# Корень проекта: при сборке в exe — папка с exe, иначе — папка с run.py
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

# При запуске из exe пути уже заданы в config по sys.executable
if getattr(sys, "frozen", False):
    import src.config as _config
    _config.PROJECT_ROOT = PROJECT_ROOT
    _config.DATA_DIR = PROJECT_ROOT / "data"
    _config.RAW_DIR = _config.DATA_DIR / "raw"
    _config.MODEL_DIR = PROJECT_ROOT / "models"
    _config.REPORTS_DIR = PROJECT_ROOT / "reports"
    _config.RESULTS_DIR = PROJECT_ROOT / "results"
    for d in (_config.DATA_DIR, _config.RAW_DIR, _config.MODEL_DIR, _config.REPORTS_DIR, _config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "evaluate"):
        print("Использование: run.py train   — обучение модели")
        print("             run.py evaluate — оценка (accuracy, F1, confusion matrix)")
        print("Данные положите в папку data/raw/ (структура: data/raw/ИмяКласса/*.jpg)")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "train":
        from src.train import main as train_main
        train_main()
    else:
        from src.evaluate import main as eval_main
        eval_main()


if __name__ == "__main__":
    main()
