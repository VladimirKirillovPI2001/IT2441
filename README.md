# Тема 1: Классификация состояния растений по изображениям

Бинарная/многоклассовая классификация (здоровое/больное растение и типы болезней) по датасету PlantVillage. Модель: CNN (EfficientNet B0).

## Цель и мотивация

Автоматическая диагностика болезней растений по фото листьев позволяет быстрее реагировать на заражение и снижать потери урожая. В проекте строится классификатор на основе предобученной CNN.

## Данные

- **Источник:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Структура: папки по классам (например `Potato___Late_blight`, `Tomato___healthy` и т.д.), внутри — изображения листьев.

## Архитектура модели

- **Модель:** EfficientNet-B0 (timm), предобученная на ImageNet.
- **Обоснование:** хороший компромисс точность/скорость, подходит для изображений среднего разрешения.

В `src/config.py` можно сменить на `resnet18` или другую модель из timm.

## Метрики качества

- Accuracy (доля верных ответов)
- F1 (macro и weighted)
- Confusion matrix (визуализация в `results/confusion_matrix.png`)

## Результаты

После обучения и запуска `python -m src.evaluate` метрики сохраняются в `results/metrics.txt`, confusion matrix — в `results/confusion_matrix.png`.

## Инструкция по запуску

### 1. Клонирование и окружение

```bash
cd project1-plant-classification
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Данные

- Скачайте [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) с Kaggle (нужен аккаунт).
- Распакуйте так, чтобы в `data/raw/` была папка с подпапками-классами (имена вида `Plant___Disease`).  
  Пример: `data/raw/PlantVillage/Potato___Late_blight/img1.jpg`, …

Либо из корня проекта (при настроенном Kaggle API):

```bash
python -m src.data.download_data
```

### 3. Обучение

Из **корня проекта**:

```bash
python -m src.train
```

Чекпоинт лучшей модели сохраняется в `models/best.pt`.

### 4. Оценка

```bash
python -m src.evaluate
```

Создаёт `results/metrics.txt` и `results/confusion_matrix.png`.

### 5. Jupyter

В папке `notebooks/` лежит ноутбук `01_explore_and_train.ipynb` для разведки данных и короткого цикла обучения.

## Релиз с exe (исполняемый файл)

1. **Сборка:** установите зависимости, затем выполните:
   ```powershell
   .\venv\Scripts\activate
   pip install pyinstaller
   pyinstaller build_exe.spec --noconfirm --clean
   ```
   Или запустите скрипт: `.\build_exe.ps1`
2. В папке `dist/PlantClassification/` появятся `PlantClassification.exe` и необходимые библиотеки.
3. Заархивируйте папку `PlantClassification` в zip.
4. На GitHub: Releases → нужный релиз → Edit → в разделе *Assets* нажмите *Attach binaries* и загрузите zip (или добавьте новый релиз и прикрепите zip).

**Запуск exe:** положите папку `data/raw/` с датасетом (подпапки-классы с картинками) рядом с exe. В консоли:
- `PlantClassification.exe train` — обучение
- `PlantClassification.exe evaluate` — оценка модели

---

## Структура репозитория

```
project1-plant-classification/
├── README.md
├── requirements.txt
├── run.py              # точка входа (train / evaluate)
├── build_exe.spec      # PyInstaller
├── build_exe.ps1       # скрипт сборки exe
├── data/
│   └── raw/          # сюда положить PlantVillage
├── models/           # best.pt после обучения
├── reports/          # history.pt
├── results/          # метрики и confusion matrix
├── notebooks/
│   └── 01_explore_and_train.ipynb
└── src/
    ├── config.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    └── data/
        ├── dataset.py
        └── download_data.py
```

## Литература / источники

- PlantVillage Dataset: [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- EfficientNet: Tan, M. & Le, Q. V. "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
- timm: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

## Лицензия

MIT.
