# DEEP-LEARNING-PROJECT

COMPANY":CODTECH IT SOLUTIONS

"NAME":T LAKSHMANNA

"Intern ID": CT06DF463

"DOMAIN": Data Science

"DURATION": 6 WEEKS

"MENTOR": NEELA SANTHOSH

# 🧠 Image Classification using CNN and TensorFlow

## 📘 Project Overview

This deep learning project demonstrates how to implement a **Convolutional Neural Network (CNN)** using **TensorFlow** to classify images from the **CIFAR-10** dataset. CNNs are widely used in computer vision tasks due to their ability to extract spatial hierarchies from images. In this project, the model is trained to recognize and classify images into 10 different categories, such as airplanes, birds, cats, and trucks. The final deliverable includes a functional model with accuracy visualization and a saved model file.

This project is ideal for internship-level applications in machine learning and deep learning, showcasing skills in data preprocessing, model building, training, evaluation, and result visualization.

---

## 🎯 Objective

The main objective of this project is to build a deep learning model capable of accurately classifying unseen image data using supervised learning. It helps demonstrate key steps involved in machine learning projects such as:

- Loading and preprocessing image data
- Designing a Convolutional Neural Network
- Training and evaluating the model
- Visualizing performance over training epochs
- Saving the trained model for deployment or future use

---

## 📊 Dataset

The project uses the **CIFAR-10** dataset, which is a standard benchmark dataset in computer vision. It contains:

- 60,000 color images (32x32 pixels)
- 50,000 images for training
- 10,000 images for testing
- 10 image classes: `Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, `Truck`

This dataset is automatically downloaded through TensorFlow’s built-in datasets API, so no manual download is needed.

---

## 🛠️ Tools & Technologies Used

- **Python 3.8+**: The programming language used for scripting
- **TensorFlow / Keras**: Framework for building and training the CNN
- **NumPy**: For handling numerical data
- **Matplotlib**: For plotting the accuracy/loss curves of the model

---

## 🧱 Model Architecture

The CNN model used in this project has the following architecture:

1. **Conv2D Layer (32 filters)**: Applies convolution using 3x3 filters
2. **MaxPooling2D Layer**: Downsamples feature maps
3. **Conv2D Layer (64 filters)**: Increases feature extraction depth
4. **MaxPooling2D Layer**
5. **Conv2D Layer (64 filters)**: Deep feature learning
6. **Flatten Layer**: Converts the 3D output to 1D for Dense layers
7. **Dense Layer (64 units, ReLU activation)**
8. **Output Layer (10 units, softmax or logits)**

This design allows the model to extract low-level and high-level features effectively and make multi-class predictions.

---

## ▶️ How to Run

Make sure you have Python installed. Then install the required libraries and run the model script.

```bash
pip install tensorflow matplotlib numpy
python cnn_image_classifier.py
