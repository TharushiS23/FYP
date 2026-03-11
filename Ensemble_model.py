import torch
from torchvision import models
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet
resnet_model = models.resnet50(weights=None)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 4)
resnet_model.load_state_dict(torch.load("FYP/resnet50_basic.pth", map_location=device))
resnet_model.to(device).eval()

# Load EfficientNet
efficient_model = models.efficientnet_b0(pretrained=False)
efficient_model.classifier[1] = torch.nn.Linear(efficient_model.classifier[1].in_features, 4)
efficient_model.load_state_dict(torch.load("FYP/efficientnetB0_basic.pth", map_location=device))
efficient_model.to(device).eval()

# Final ensemble weights
resnet_weights = torch.tensor([0.6, 0.6, 0.6, 0.6]).to(device)
effnet_weights = 1 - resnet_weights

# preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class_names = ["512Glioma","512Meningioma","512Normal","512Pituitary"]


def predict(image_path):

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        r_logits = resnet_model(img_tensor)
        e_logits = efficient_model(img_tensor)

        weighted_logits = r_logits * resnet_weights + e_logits * effnet_weights
        probs = torch.softmax(weighted_logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return class_names[pred]