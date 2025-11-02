from typing import Tuple
import torch
import torch.nn as nn
from PIL import Image
import open_clip


def create_clip_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        train_only_last_layer: bool = False,
        model_name: str = "ViT-B-16",
        pretrained_dataset: str = "openai"
) -> Tuple[nn.Module, object]:
    # Загружаем CLIP backbone через open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_dataset if pretrained else None
    )

    # Визуальный backbone
    visual_encoder = model.visual
    in_features = visual_encoder.output_dim

    # Классификационная голова
    classifier = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(classifier.weight, std=0.02)
    if classifier.bias is not None:
        nn.init.zeros_(classifier.bias)

    # Собираем в Sequential: (0) encoder + (1) head
    model = nn.Sequential(visual_encoder, classifier)

    # Замораживаем энкодер, если нужно
    if train_only_last_layer:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("1")

    return model, preprocess


if __name__ == "__main__":
    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert("RGB")

    model, preprocess = create_clip_classifier(
        num_classes=2,
        pretrained=True,
        train_only_last_layer=True,  # только последний слой
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
