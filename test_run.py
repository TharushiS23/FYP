import os
from PIL import Image
import torch
from torchvision import transforms

# ---- paths inside your repo ----
MODEL_PATH = "models/ensemble_model.pth"
IMAGE_PATH = "test_images/sample.jpg"

# ---- simple checks ----
print("Model exists:", os.path.exists(MODEL_PATH))
print("Image exists:", os.path.exists(IMAGE_PATH))

# ---- load image ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0)

print("Image tensor shape:", image.shape)

# ---- load model file only ----
model_data = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
print("Model file loaded successfully")
print(type(model_data))