"""CNN model for plant classification (EfficientNet / ResNet via timm)."""
import torch.nn as nn
import timm
from src.config import MODEL_NAME


def build_model(num_classes: int, pretrained: bool = True):
    """Build classifier. MODEL_NAME from config: efficientnet_b0, resnet18, etc."""
    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
