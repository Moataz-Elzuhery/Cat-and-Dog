# 🐱🐶 Cat vs Dog Image Classifier (CNN)

A deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 🚀 Overview

This project implements an end-to-end deep learning pipeline for image classification:

- Data preprocessing  
- Model training  
- Model evaluation  
- Prediction on new images  

The model is trained on a labeled dataset of cats and dogs and achieves strong performance on unseen data.

---

## 🧠 Model Details

- Architecture: Convolutional Neural Network (CNN)  
- Framework: TensorFlow / Keras  
- Input size: 150 × 150 images  
- Output: Binary classification (Cat 🐱 / Dog 🐶)  

---

## 📊 Results

- Training Accuracy: ~94%  
- Validation Accuracy: **83.4%**  

---

## ⚙️ Features

- Image preprocessing and normalization  
- Train/validation split using ImageDataGenerator  
- ModelCheckpoint to save the best model  
- Real-time prediction on new images  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
python train.py
python predict.py
