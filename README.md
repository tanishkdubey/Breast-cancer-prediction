# 🧠 Breast Cancer Prediction with ANN

Predicts whether a tumor is malignant or benign using an Artificial Neural Network (ANN) trained on the Breast Cancer Wisconsin dataset. The model uses dropout and early stopping for regularization and achieves 97.37% validation accuracy.

---

## 📌 Project Highlights

| Component             | Description                                               |
|----------------------|-----------------------------------------------------------|
| 📊 Model Type         | Feed-forward ANN (Keras Sequential)                       |
| 🎯 Objective          | Binary classification: Malignant (1) or Benign (0)        |
| 💂️ Dataset            | Breast Cancer Wisconsin Dataset (from sklearn)           |
| 🧪 Validation Accuracy | 97.37% (with early stopping and dropout regularization)   |
| ⚙️ Techniques Used    | Dropout, EarlyStopping, StandardScaler                    |

---

## 🧬 Dataset Overview

- 569 total samples
- 30 numerical features per sample
- 2 classes:
  - 0 → Benign
  - 1 → Malignant

📚 Source: Built into scikit-learn → load_breast_cancer()

---

## 🧠 Model Architecture

```
Input (31 features)
↓
Dense(64, relu)
↓
Dense(32, relu)
↓
Dropout(0.2)
↓
Dense(1, sigmoid)
```

- Loss Function: Binary Crossentropy  
- Optimizer: Adam  
- Evaluation Metric: Accuracy  
- EarlyStopping: Patience = 5 (monitor = val_loss)  
- Dropout: Prevents overfitting by randomly deactivating neurons during training

---

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:

```bash
pip install numpy scikit-learn tensorflow
```

3. Run the script:

```bash
python ann_breast_cancer.py
```

4. Output: Console will show training/validation loss and final accuracy

---

## 📈 Results

| Metric               | Value         |
|----------------------|---------------|
| Validation Accuracy  | 97.37%        |
| Loss Function        | Binary Crossentropy |
| Regularization Used  | Dropout + EarlyStopping |

---

## 🛠 Future Improvements

- Hyperparameter tuning (layers, units, dropout rate)
- SHAP or LIME for feature importance
- Model checkpointing and persistence
- Deploy as a Streamlit web app

---

© 2025 Tanishk Dubey — Feel free to fork and contribute!

