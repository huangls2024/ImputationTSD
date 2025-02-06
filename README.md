
# 📌 Improving Sensing Integrity in Thermal Power Plants via a Hybrid Multi-Head Attention Model

This repository provides the **source code** for our paper:

> **L. Huang et al., "Improving Sensing Integrity in Thermal Power Plants via a Hybrid Multi-Head Attention Model."**

---


---

## 📊 Dataset Description

The dataset was collected from a **coal-fired thermal power plant** in **Shandong, China**, containing **131,040 samples** recorded from **2020 to 2021** \raw_data.csv. Each sample includes:

- 🌡 **Ambient temperature**
- 🔥 **Main steam temperature**
- ⚙ **Main steam pressure**
- 🔄 **Reheat steam temperature**
- 💨 **Exhaust gas temperature**
- 💦 **Feed water temperature**
- 🏭 **Back pressure**
- 🧪 **Oxygen content**

---

## 🔍 Baseline Methods

We evaluate **two categories** of imputation methods:

### 1️⃣ **Non-learning-based methods**
   - 📈 Statistical interpolation
   - 🔄 Traditional time-series imputation

### 2️⃣ **Learning-based methods**
   - 🤖 Deep learning-based sequence models
   - 🔬 Transformer-based architectures

We also refer to **PyPOTS**, an open-source Python toolbox for **partial observed time-series learning**, in our implementation of imputation baselines.

📌 **Reference Project**: [PyPOTS - Wenjie Du](https://github.com/WenjieDu/PyPOTS)  

---
---

## 🚀 HybridMHA Model

The **Hybrid Multi-Head Attention (HybridMHA) model** is designed to perform **sensor data imputation** in thermal power plants.

- **Input**: Time-series data with missing values.
- **Output**: Fully imputed time-series data.
- **Architecture**: Combines **Dynamic Sparse Attention** and **Diagonally-Masked Attention**.

📌 **Source Code**: [`imputation/HybridMHA`](./imputation/HybridMHA)  

📊 **Model Diagram** *(as shown in the paper)*:  
 
