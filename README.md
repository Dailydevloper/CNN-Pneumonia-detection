# ✅ Pneumonia Detection Using Deep Learning (CNN + Transfer Learning)

## 📌 Overview

This project is an end-to-end deep learning system for detecting Pneumonia from Chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning (MobileNetV2).

The system is trained on a medical imaging dataset, optimized for generalization, and deployed as a web application using FastAPI.

Users can upload an X-ray image and instantly get a prediction with confidence score.

## 🎯 Features

✅ Chest X-ray classification (NORMAL vs PNEUMONIA)

✅ Transfer Learning using MobileNetV2

✅ Data augmentation for better generalization

✅ Overfitting control (regularization + early stopping)

✅ Threshold tuning for balanced predictions

✅ FastAPI-based web interface

✅ Cloud-ready deployment setup

✅ Confidence score in predictions

## 🗂️ Project Structure

```
pneumonia_detector/
│
├── data/                   # Dataset (ignored in GitHub)
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebooks/              # Training & experimentation notebooks
│   ├── pneumonia_train_clean.ipynb
│   └── 01_baseline_cnn.ipynb
│
├── deploy/                 # Web application
│   ├── app.py
│   ├── requirements.txt
│   ├── model/
│   │   └── pneumonia_model.keras
│   ├── templates/
│   └── static/
│
├── src/                    # Utility scripts
│   └── split_data.py
│
├── models/                 # Saved experimental models
│
├── .gitignore
└── README.md
```

## 📊 Dataset

This project uses the Chest X-Ray Pneumonia Dataset from Kaggle.

🔗 Link:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Dataset Structure

After downloading and extracting:

```
data/
 ├── train/
 │   ├── NORMAL/
 │   └── PNEUMONIA/
 ├── val/
 │   ├── NORMAL/
 │   └── PNEUMONIA/
 └── test/
     ├── NORMAL/
     └── PNEUMONIA/
```

⚠️ The dataset is not uploaded to GitHub due to size and licensing restrictions.

## 🧠 Model Architecture

**Base Model:** MobileNetV2 (Pretrained on ImageNet)

- Fine-tuned last layers
- Global Average Pooling
- Fully Connected Layer (128 neurons)
- Dropout (0.4)
- Output: Sigmoid (Binary Classification)

### Training Techniques

- Transfer Learning
- Data Augmentation
- L2 Regularization
- Early Stopping
- Learning Rate = 1e-5
- Binary Crossentropy Loss

## 📈 Model Performance

### Final Evaluation

| Metric         | Value |
| -------------- | ----- |
| Validation Acc | ~93%  |
| Test Acc       | ~84%  |

### Observations

- High pneumonia recall
- Balanced predictions after threshold tuning
- Reduced overfitting

## 🛠️ Tech Stack

| Category     | Tools              |
| ------------ | ------------------ |
| Language     | Python             |
| DL Framework | TensorFlow / Keras |
| Model        | MobileNetV2        |
| Backend      | FastAPI            |
| Frontend     | HTML, CSS          |
| Deployment   | Render             |
| Tools        | Git, Jupyter       |

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Dailydevloper/CNN-Pneumonia-detection.git
cd CNN-Pneumonia-detection
```

### 2️⃣ Create Environment

```bash
conda create -n tf python=3.10
conda activate tf
```

### 3️⃣ Install Dependencies

```bash
cd deploy
pip install -r requirements.txt
```

### 4️⃣ Download Dataset

Download from Kaggle and extract into:

```
data/
```

## 🚀 Training the Model

Open Jupyter:

```bash
python -m jupyter lab
```

Run:

```
notebooks/pneumonia_train_clean.ipynb
```

This notebook covers:

- Data loading
- Augmentation
- Training
- Evaluation
- Saving model

## 🌐 Running the Web App (Local)

```bash
cd deploy
python -m uvicorn app:app --reload
```

Open browser:

```
http://127.0.0.1:8000
```

Upload an X-ray image to test.

## ☁️ Cloud Deployment (Render)

This project is ready for deployment on Render.

### Render Config

| Setting  | Value                                       |
| -------- | ------------------------------------------- |
| Root Dir | deploy                                      |
| Build    | pip install -r requirements.txt             |
| Start    | uvicorn app:app --host 0.0.0.0 --port 10000 |

## 📌 Prediction Logic

The model outputs a probability score.

Custom threshold:

```
THRESHOLD = 0.85
```

Decision:

- **> 0.85** → Pneumonia
- **≤ 0.85** → Normal

This improves balance and reduces false positives.

## ⚠️ Disclaimer

This system is intended for educational and research purposes only.

It is **NOT** a medical diagnostic tool and should not be used for clinical decision-making.

Always consult certified medical professionals.

## 📷 Sample Output

```
Result: NORMAL ✅ (34.9%)
Result: PNEUMONIA ⚠️ (99.8%)
```

## 📌 Future Improvements

- Grad-CAM explainability
- Multi-class disease detection
- TF Lite mobile deployment
- Cloud GPU inference
- Model ensemble
- Clinical dataset validation

## 👨‍💻 Author

**Prateek Dwivedi**

B.Tech Student | AI & ML Enthusiast

📫 GitHub: https://github.com/Dailydevloper

## ⭐ Acknowledgements

- Kaggle Dataset Contributors
- TensorFlow Team
- Open Source Community

---

If you find this project useful, feel free to ⭐ the repository!
