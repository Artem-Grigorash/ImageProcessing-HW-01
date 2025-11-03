from typing import Tuple
import torch
import torch.nn as nn
from PIL import Image
import open_clip


def freeze_partial_layers(model: nn.Module, trainable_ratio: float = 0.3):
    """
    Размораживает последние X% слоёв модели.
    trainable_ratio=0.3 → обучаем последние 30% параметров.
    """
    all_params = list(model.named_parameters())
    total = len(all_params)
    cutoff = int(total * (1 - trainable_ratio))
    for i, (_, p) in enumerate(all_params):
        p.requires_grad = i >= cutoff

    print(f"✅ Fine-tuning {trainable_ratio * 100:.1f}% последних слоёв "
          f"({total - cutoff}/{total})")


def create_clip_partial_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        trainable_ratio: float = 0.3,
        model_name: str = "ViT-B-16",
        pretrained_dataset: str = "openai"
) -> Tuple[nn.Module, object]:
    """
    Создаёт CLIP визуальный энкодер и размораживает только часть последних слоёв.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_dataset if pretrained else None
    )

    visual_encoder = model.visual
    in_features = visual_encoder.output_dim

    classifier = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(classifier.weight, std=0.02)
    if classifier.bias is not None:
        nn.init.zeros_(classifier.bias)

    model = nn.Sequential(visual_encoder, classifier)

    freeze_partial_layers(model, trainable_ratio=trainable_ratio)
    return model, preprocess


if __name__ == "__main__":
    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert("RGB")

    # Fine-tune последних 30 % CLIP-энкодера
    model, preprocess = create_clip_partial_classifier(
        num_classes=2,
        pretrained=True,
        trainable_ratio=0.3,
        model_name="ViT-B-16",
        pretrained_dataset="openai"
    )

    model.eval()

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    pred_class = probs.argmax().item()
    conf = probs[pred_class].item()

    print(f"🧠 Модель: CLIP {model[0].__class__.__name__}")
    print(f"Предсказанный класс: {pred_class}")
    print(f"Уверенность: {conf:.4f}")
    print(f"Вероятности: {probs.tolist()}")
