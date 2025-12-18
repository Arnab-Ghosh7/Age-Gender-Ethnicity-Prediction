# 👤 Age, Gender & Ethnicity Prediction using Deep Learning

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---
## 📌 Project Overview

Understanding human demographic attributes from facial images is a key problem in the fields of **computer vision** and **artificial intelligence**. This project presents a **deep learning–based system** capable of predicting **Age, Gender, and Ethnicity** from facial image data using **Convolutional Neural Networks (CNNs)**.

The project is designed to simulate a **real-world machine learning workflow**, starting from raw pixel-level data and progressing through preprocessing, exploratory data analysis, model training, and performance evaluation. Instead of relying on pre-processed images, the dataset represents facial information as **pixel arrays**, requiring careful normalization, reshaping, and visualization before model training.

A multi-task learning approach is adopted, where a shared CNN backbone learns common facial features, while separate output heads specialize in predicting age (regression), gender (binary classification), and ethnicity (multi-class classification). This approach improves learning efficiency and better reflects practical AI system design.

Special emphasis is placed on:
- Handling imbalanced demographic data  
- Visualizing data distributions to guide modeling decisions  
- Building modular, interpretable, and reproducible code  
- Ensuring compatibility with Kaggle notebook environments  

The final outcome is a **scalable and extensible demographic prediction system** that demonstrates both **deep learning proficiency** and **applied machine learning engineering skills**, making it suitable for academic exploration as well as industry-oriented applications.

This project serves as a strong foundation for future extensions such as **transfer learning**, **real-time inference**, and **deployment as a web application**.


---

## 🎯 Objectives

- Predict **Age** as a regression task  
- Predict **Gender** as a binary classification task  
- Predict **Ethnicity** as a multi-class classification task  
- Preprocess raw pixel data into CNN-compatible format  
- Visualize data distributions and class imbalance  
- Train and evaluate deep learning models  

---

## 🧠 Dataset Description

The dataset consists of facial images represented as:
- `pixels` → NumPy arrays (grayscale image data)
- `age` → Integer values
- `gender` → Binary labels (0: Male, 1: Female)
- `ethnicity` → Categorical labels

### Data Characteristics
- Images are grayscale
- Pixel values normalized to range **[0, 1]**
- Multiple labels per image (multi-task learning friendly)

---

## 🔄 Project Workflow

1. **Data Loading**
   - Load dataset from Kaggle
   - Inspect data types and distributions

2. **Data Preprocessing**
   - Normalize pixel values
   - Reshape pixel arrays into image format
   - Handle class imbalance

3. **Exploratory Data Analysis**
   - Age distribution visualization
   - Gender distribution visualization
   - Ethnicity distribution visualization

4. **Model Building**
   - CNN architecture for feature extraction
   - Separate heads for age, gender, and ethnicity prediction

5. **Model Training**
   - Loss functions suited for each task
   - Optimization using backpropagation

6. **Evaluation**
   - Accuracy for gender & ethnicity
   - Error metrics for age prediction

---

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Plotly  
- **Deep Learning:** TensorFlow / Keras  
- **Visualization:** Plotly, Matplotlib  
- **Platform:** Kaggle Notebook  

---

## 📊 Data Visualization

The project includes interactive visualizations for:
- Age distribution
- Gender distribution
- Ethnicity distribution

These visualizations help in:
- Understanding dataset imbalance
- Designing better loss functions
- Improving model generalization

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/Age-Gender-Ethnicity-Prediction.git
cd Age-Gender-Ethnicity-Prediction
```
## ▶️ How to Run

- Open Kaggle Notebook

- Attach the dataset ()

- Run notebook

## 👤 Author

Arnab Ghosh
