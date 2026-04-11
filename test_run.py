import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

st.title("Repo File Test")

MODEL_PATH = "efficientnetB0_basic.pth"
IMAGE_PATH = r"C:\Users\tharu\OneDrive\Documents\IIT\IIT- Fourth year\Tharushi\FYP - Tharu\FYP\FYP\Split\test\512Meningioma\M_4.jpg"

st.write("Model exists:", os.path.exists(MODEL_PATH))
st.write("Image exists:", os.path.exists(IMAGE_PATH))

if os.path.exists(IMAGE_PATH):
    image = Image.open(IMAGE_PATH).convert("RGB")
    st.image(image, caption="Test image", width=200)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)
    st.write("Image tensor shape:", tuple(img_tensor.shape))

if os.path.exists(MODEL_PATH):
    model_data = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    st.success("Model file loaded successfully")
    st.write("Loaded object type:", str(type(model_data)))