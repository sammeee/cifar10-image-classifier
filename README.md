# CIFAR-10 Image Classifier

[![Hugging Face Space](https://img.shields.io/badge/🚀-Live_Demo-blue)](https://huggingface.co/spaces/szamadesh/cifar10-classifier)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

A clean, interactive web application that classifies images into the 10 CIFAR-10 categories using a **fine-tuned ResNet-18** model.

## 🎯 Live Demo

Try the live application here → **[CIFAR-10 Classifier on Hugging Face Spaces](https://huggingface.co/spaces/szamadesh/cifar10-classifier)**

## 📋 Project Overview

This project demonstrates a complete end-to-end computer vision workflow:
- Fine-tuning a pre-trained **ResNet-18** model on the CIFAR-10 dataset
- Building a user-friendly web interface with **Streamlit**
- Deploying the model as a public web app using **Hugging Face Spaces** (Docker-based)

### Key Features
- Image upload with real-time prediction
- Displays top 3 predictions with confidence scores
- Clean, responsive, and mobile-friendly UI
- Model caching for fast inference

## 🛠️ Technologies Used
- **PyTorch** + torchvision (ResNet-18)
- **Streamlit** for the web interface
- **Hugging Face Spaces** (Docker) for deployment
- Docker

## 📁 Project Structure

```bash
cifar10-image-classifier/
├── src/
│   └── streamlit_app.py     # Main Streamlit application
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore

**Note**: The trained model file (`tuned_resnet_cifar10.pth` ≈ 44.8 MB) is not included in this repository due to GitHub's file size limit. The full model is loaded and running in the live [Hugging Face Space](your-link-here).

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash 
   git clone https://github.com/sammeee/cifar10-image-classifier.git
   cd cifar10-image-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash 
   streamlit run src/streamlit_app.py
   ```

---

## 📊 Results & Observations

*   **Test Accuracy**: ~92-94% (High performance for a lightweight ResNet-18 on 32×32 images).
*   **Strengths**: Performs well on clear, centered images.
*   **Limitations**: Common confusions occur between visually similar classes (e.g., dog ↔ deer, cat ↔ dog). This is expected due to the low resolution (32x32) of the CIFAR-10 dataset.

---

## 🔮 Future Improvements

- [ ] Add data augmentation during training.
- [ ] Experiment with larger backbones (ResNet-34, EfficientNet-B0, etc.).
- [ ] Add Grad-CAM visualization for model explainability.
- [ ] Deploy with a FastAPI backend + Streamlit frontend for better scalability.

---

## 📄 License

Distributed under the **Apache 2.0 License**.

---

*Built as part of my self-taught Deep Learning journey using PyTorch and Streamlit. Feel free to explore the live demo and provide feedback!*
