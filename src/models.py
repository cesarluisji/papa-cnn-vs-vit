"""Modelos para clasificación de enfermedades en papa, café y cacao.

Tres arquitecturas, todas con la misma interfaz: imagen RGB 224x224 -> logits (num_classes).

- cnn_small : CNN pequeña entrenada desde cero (línea base neuronal, R3).
- resnet50  : CNN residual (He et al., 2016), preentrenada en ImageNet o desde cero.
- vit_small : Vision Transformer ViT-S/16 (Dosovitskiy et al., 2021), preentrenado o desde cero.

El flag `pretrained` permite la ablación preentrenado vs. desde cero (R4).
"""
import torch
import torch.nn as nn
import timm

# Clases por cultivo (según los artículos de cada dataset)
CLASES = {
    "papa": ["bacteria", "fungi", "healthy", "nematode", "pest", "phytophthora", "virus"],
    "cafe": ["healthy", "red_spider_mite", "rust_1", "rust_2", "rust_3", "rust_4"],
    "cacao": ["h0", "m1", "m2", "m3"],
}


class CNNSmall(nn.Module):
    """4 bloques Conv-BN-ReLU-MaxPool + pooling global + clasificador lineal."""

    def __init__(self, num_classes: int, widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for c_out in widths:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


TIMM_NAMES = {
    "resnet50": "resnet50",
    "vit_small": "vit_small_patch16_224",
}


def build_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    """Construye el modelo. Para timm, reemplaza la cabeza por una de `num_classes` salidas."""
    if name == "cnn_small":
        return CNNSmall(num_classes)
    if name in TIMM_NAMES:
        return timm.create_model(
            TIMM_NAMES[name], pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
        )
    raise ValueError(f"Modelo desconocido: {name}. Opciones: cnn_small, {', '.join(TIMM_NAMES)}")


def gradcam_target_layer(model: nn.Module, name: str):
    """Capa sobre la que se calcula Grad-CAM (R5)."""
    if name == "cnn_small":
        return model.features[-4]          # última conv
    if name == "resnet50":
        return model.layer4[-1]            # último bloque residual
    if name == "vit_small":
        return model.blocks[-1].norm1      # último bloque Transformer (tokens -> mapa 14x14)
    raise ValueError(name)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
