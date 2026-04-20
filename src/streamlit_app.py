import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import os

# Page config
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="📸")

# Title and description
st.title("📸 CIFAR-10 Image Classifier")
st.markdown("Upload an image and the model will predict which of the 10 classes it belongs to.")

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
        
        # Safe path that works in HF Spaces Docker template
        model_path = os.path.join(os.path.dirname(__file__), "tuned_resnet_cifar10.pth")
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# Correct CIFAR-10 normalization (very important!)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Classifying..."):
            img_tensor = transform(image).unsqueeze(0)   # add batch dimension
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item() * 100
            
        st.subheader("✅ Prediction")
        st.success(f"**Class:** {classes[predicted_idx]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        # Top 3 predictions
        st.subheader("Top 3 Predictions")
        top3 = torch.topk(probabilities, 3)
        for i in range(3):
            idx = top3.indices[i].item()
            prob = top3.values[i].item() * 100
            st.write(f"{classes[idx]}: **{prob:.2f}%**")
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.exception(e)   # Shows full error for debugging