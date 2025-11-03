from typing import Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ViT_B_16_Weights  # type: ignore


def _get_default_weights(pretrained: bool):
    return ViT_B_16_Weights.DEFAULT if pretrained else None


def freeze_partial_layers(model: nn.Module, trainable_ratio: float = 0.3):
    """
    Размораживает только часть последних слоёв модели.
    trainable_ratio=0.3 -> обучаем 30% последних слоёв.
    """
    all_layers = list(model.named_parameters())
    total = len(all_layers)
    cutoff = int(total * (1 - trainable_ratio))

    for i, (_, p) in enumerate(all_layers):
        p.requires_grad = i >= cutoff

    print(f"✅ Fine-tuning {trainable_ratio * 100:.1f}% последних слоёв "
          f"({total - cutoff} из {total})")


def create_vit_partial_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        trainable_ratio: float = 0.3,
        weights: Optional[object] = None,
) -> Tuple[nn.Module, Optional[object]]:
    """
    Создаёт ViT и размораживает только часть последних слоёв.
    """
    used_weights = weights if weights is not None else _get_default_weights(pretrained)
    model = models.vit_b_16(weights=used_weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
    if model.heads.head.bias is not None:
        nn.init.zeros_(model.heads.head.bias)

    freeze_partial_layers(model, trainable_ratio=trainable_ratio)
    return model, used_weights


def load_partial_vit():
    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert('RGB')

    # Создаём ViT, обучаем последние 30 % слоёв
    model, weights = create_vit_partial_classifier(
        num_classes=2,
        pretrained=True,
        trainable_ratio=0.3
    )
    model.eval()

    preprocess = weights.transforms()
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    pred = torch.argmax(probs).item()
    conf = probs[pred].item()

    print(f"Image: {image_path}")
    print(f"Predicted class: {pred}")
    print(f"Confidence: {conf:.4f}")
    print(f"Class probabilities: {probs.tolist()}")


if __name__ == '__main__':
    load_partial_vit()
