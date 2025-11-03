from typing import Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import Swin_T_Weights


def _get_default_weights(pretrained: bool):
    return Swin_T_Weights.DEFAULT if pretrained else None


def freeze_partial_layers(model: nn.Module, trainable_ratio: float = 0.3):
    """
    Замораживает верхние (1 - trainable_ratio) слоёв модели.
    Например, trainable_ratio=0.3 → 30% последних слоёв остаются обучаемыми.
    """
    all_layers = list(model.named_parameters())
    total_layers = len(all_layers)
    cutoff = int(total_layers * (1 - trainable_ratio))

    for i, (name, param) in enumerate(all_layers):
        param.requires_grad = i >= cutoff

    print(f"✅ Разморожено {total_layers - cutoff} из {total_layers} слоёв "
          f"({trainable_ratio * 100:.1f}% модели).")


def create_swin_partial_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        trainable_ratio: float = 0.3,
        weights: Optional[object] = None,
) -> Tuple[nn.Module, Optional[object]]:
    """
    Создаёт Swin и размораживает только часть последних слоёв.
    """
    used_weights = weights if weights is not None else _get_default_weights(pretrained)
    model = models.swin_t(weights=used_weights)

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    # Замораживаем часть параметров (fine-tune только последние X%)
    freeze_partial_layers(model, trainable_ratio=trainable_ratio)

    return model, used_weights


if __name__ == '__main__':
    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert('RGB')

    model, weights = create_swin_partial_classifier(
        num_classes=2,
        pretrained=True,
        trainable_ratio=0.3  # ← обучаем последние 30%
    )

    model.eval()

    preprocess = weights.transforms()
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    pred_class = probs.argmax().item()
    conf = probs[pred_class].item()

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {conf:.4f}")
