import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load tuned ResNet50 model
# -----------------------------
resnet_model_tuned_ensemble = models.resnet50(weights=None)
resnet_model_tuned_ensemble.fc = nn.Linear(
    resnet_model_tuned_ensemble.fc.in_features, 4
)
resnet_model_tuned_ensemble.load_state_dict(
    torch.load("FYP/resnet50_finetuned_optuna.pth", map_location=device)
)
resnet_model_tuned_ensemble = resnet_model_tuned_ensemble.to(device)
resnet_model_tuned_ensemble.eval()

# -----------------------------
# Load tuned EfficientNetB0 model
# -----------------------------
efficient_model_tuned_ensemble = models.efficientnet_b0(weights=None)
efficient_model_tuned_ensemble.classifier[1] = nn.Linear(
    efficient_model_tuned_ensemble.classifier[1].in_features, 4
)
efficient_model_tuned_ensemble.load_state_dict(
    torch.load("FYP/efficientnetB0_finetuned_optuna.pth", map_location=device)
)
efficient_model_tuned_ensemble = efficient_model_tuned_ensemble.to(device)
efficient_model_tuned_ensemble.eval()

# -----------------------------
# Final tuned ensemble weights
# -----------------------------
resnet_class_weights_tuned = torch.tensor(
    [0.4, 0.4, 0.6, 0.6], dtype=torch.float32
).to(device)

effnet_class_weights_tuned = 1.0 - resnet_class_weights_tuned

# -----------------------------
# Preprocessing
# -----------------------------
preprocess_tuned_ensemble = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Class names
# Must match the training class index order
# -----------------------------
class_names_tuned_ensemble = [
    "512Glioma",
    "512Meningioma",
    "512Normal",
    "512Pituitary"
]

# -----------------------------
# Prediction function
# -----------------------------
def predict_tuned_ensemble(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess_tuned_ensemble(img).unsqueeze(0).to(device)

    with torch.no_grad():
        resnet_logits_tuned = resnet_model_tuned_ensemble(img_tensor)
        effnet_logits_tuned = efficient_model_tuned_ensemble(img_tensor)

        weighted_logits_tuned = (
            resnet_logits_tuned * resnet_class_weights_tuned
            + effnet_logits_tuned * effnet_class_weights_tuned
        )

        probabilities_tuned = torch.softmax(weighted_logits_tuned, dim=1)
        predicted_index_tuned = torch.argmax(probabilities_tuned, dim=1).item()

    return class_names_tuned_ensemble[predicted_index_tuned]