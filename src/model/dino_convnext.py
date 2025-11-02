import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


class DinoV3SwinClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, train_only_last_layer: bool = False):
        super().__init__()
        self.num_labels = num_labels

        # Загружаем предобученную модель DINOv3
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.base_model.num_features

        # Классификационная голова
        self.classifier_head = nn.Linear(in_features, num_labels)

        # Замораживаем весь backbone, если указано
        if train_only_last_layer:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.base_model(x)
        logits = self.classifier_head(features)
        return logits


def create_dino_swin_classifier(
        num_classes: int = 2,
        pretrained: bool = True,
        train_only_last_layer: bool = False,
        model_name: str = "vit_base_patch16_dinov3"
) -> DinoV3SwinClassifier:
    model = DinoV3SwinClassifier(model_name, num_classes, train_only_last_layer)
    return model


def load_dinoV3_swin():
    model_name = "vit_base_patch16_dinov3"
    image_path = "data/mac-merged/0.png"

    # Препроцессинг картинки под timm
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Файл не найден: {image_path}")
        return

    image_tensor = transform(image).unsqueeze(0)

    # Создаём модель: backbone заморожен, учится только последний слой
    model = create_dino_swin_classifier(
        num_classes=2,
        pretrained=True,
        train_only_last_layer=True,
        model_name=model_name
    )

    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

    print(f"🔥 Модель: {model_name}")
    print(f"Логиты: {logits.numpy()}")
    print(f"Вероятности: {probs.numpy()}")
    print(f"Предсказанный класс: {pred_class}")


if __name__ == "__main__":
    load_dinoV3_swin()
