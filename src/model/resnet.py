from typing import Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet152_Weights  # type: ignore


def _get_default_weights(pretrained: bool):
    if pretrained:
        return ResNet50_Weights.DEFAULT
    return None

def load_resnet_from_weights(weights_path: str, num_classes: int) -> nn.Module:
    model, _ = create_resnet_classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model

def create_resnet_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        train_only_last_layer: bool = False,
        weights: Optional[object] = None,
) -> Tuple[nn.Module, Optional[object]]:
    used_weights = weights if weights is not None else _get_default_weights(pretrained)
    model = models.resnet50(weights=used_weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.fc.weight, std=0.02)
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias)

    if train_only_last_layer:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")

    return model, used_weights



if __name__ == '__main__':
    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert('RGB')

    model, weights = create_resnet_classifier(num_classes=2, pretrained=True, freeze_backbone=True)
    model.eval()

    preprocess = weights.transforms()

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

    print(f"Image: {image_path}")
    print(f"Image size: {image.size}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class probabilities: {probabilities.tolist()}")
