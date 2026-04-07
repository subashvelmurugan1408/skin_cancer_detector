# 🧠 Skin Cancer Detection using Deep Learning

A web-based AI application that detects whether a skin lesion is **benign or malignant** using a Convolutional Neural Network (CNN).

---

## 🚀 Live Demo

🔗 *Add your deployed app link here*

---

## 📌 Features

* 📤 Upload skin images (JPG/PNG)
* 🤖 AI-powered prediction (Benign / Malignant)
* 📊 Confidence score display
* 🎨 Premium UI with Streamlit
* ⚡ Fast and lightweight model

---

## 🧠 Model Details

* Model Type: Convolutional Neural Network (CNN)
* Architecture:

  * Conv2D → MaxPooling
  * Conv2D → MaxPooling
  * Conv2D → MaxPooling
  * Flatten → Dense → Dropout → Output
* Input Size: 224x224
* Output: Binary Classification (Benign / Malignant)

---

## 📂 Dataset

* HAM10000 Dataset
* Additional skin cancer dataset (merged for better balance)

### ⚖️ Data Improvement

* Handled class imbalance
* Increased malignant samples
* Applied data augmentation

---

## 🛠️ Tech Stack

* Python 🐍
* TensorFlow / Keras
* OpenCV
* Streamlit

---

## 📸 Screenshots

*(Add screenshots of your app here)*

---

## ⚠️ Disclaimer

This project is for educational purposes only.
It is **not a medical diagnosis tool**. Always consult a healthcare professional.

---

## 🧑‍💻 Author

**Subash**

---

## ⭐ Acknowledgements

* Kaggle datasets
* Open-source ML community

---

## 📦 Installation

```bash
git clone https://github.com/your-username/skin-cancer-detector.git
cd skin-cancer-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Future Improvements

* Multi-class classification
* Explainable AI (Grad-CAM)
* Mobile app integration
* Cloud deployment optimization

---
