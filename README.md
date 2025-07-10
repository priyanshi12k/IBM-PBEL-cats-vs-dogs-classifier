# 🐱 Cats vs Dogs Image Classifier 🐶

This project classifies images of **cats** and **dogs** using a **Convolutional Neural Network (CNN)** built with **Keras**. The model is trained in a Jupyter notebook and deployed as a Streamlit web app on **Hugging Face Spaces**.

---

## 🚀 Live Demo

👉 [**Click here to try the app**](https://huggingface.co/spaces/priyanshi12k/cats-vs-dogs-classifier)

Upload, paste, or link an image — get real-time predictions with confidence scores.

---

## 🗂 Files in This Repository

| File | Description |
|------|-------------|
| `Cats_vs_Dogs_Classification.ipynb` | Jupyter Notebook for training the CNN model |
| `app.py` | Streamlit deployment script (runs the web app) |
| `requirements.txt` | Python dependencies needed for deployment |
| `README.md` | Overview of the project |

> 🔒 Note: The trained model file (`best_model.keras`) is **not included** in this repo. It is hosted within the Hugging Face Space.

---

## 🧠 Model Overview

- **Architecture**: CNN with Conv2D + MaxPooling layers, followed by Dense layers
- **Input Shape**: 256x256x3
- **Output**: Sigmoid activation for binary classification
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

---

## 📊 Model Performance

| Metric        | Score        |
|---------------|--------------|
| Validation Accuracy | ~85% |
| Precision (macro)   | 85%  |
| Recall (macro)      | 84%  |
| F1 Score (macro)    | 84%  |
| Confusion Matrix    | Included in the notebook |

---

## 🧪 Dataset

- **Name**: Dogs vs Cats
- **Source**: [Kaggle: Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Size Used**: ~25,000 images
- **Classes**: 0 = Cat, 1 = Dog

---

## 🖥️ Web App Features

- ✅ Upload images from your device (drag, browse, or paste)
- ✅ Paste an image URL
- ✅ Predicts **Cat 🐱** or **Dog 🐶**
- ✅ Shows prediction confidence as a progress bar
- ✅ User-friendly, responsive UI

---

## 🛠 Built With

- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow, Requests
- Hugging Face Spaces

---

## 🔧 Run the Web App Locally

### 📦 Prerequisites
- Python 3.8+
- Download or train a model and save it as `best_model.keras`

### 🧪 Instructions

```bash
git clone https://github.com/priyanshi12k/IBM-PBEL-cats-vs-dogs-classifier
cd IBM-PBEL-cats-vs-dogs-classifier
pip install -r requirements.txt
streamlit run app.py
