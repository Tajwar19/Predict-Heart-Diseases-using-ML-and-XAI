
# 💓 Heart Disease Detection using LightGBM and SHAP

This project applies a machine learning approach to predict the presence of heart disease based on patient clinical attributes. It uses the LightGBM classifier with randomized hyperparameter tuning and SHAP explainability to deliver accurate and interpretable results.

---

## 🚀 Project Highlights

- ✅ **Model**: LightGBM (with RandomizedSearchCV for hyperparameter tuning)
- 📊 **Accuracy**: 100% on test set
- 🔍 **Explainability**: SHAP (both global and local)
- 🧠 **Dataset**: Standard heart disease dataset with 13 clinical features
- 💾 **Model**: Exported as a full pipeline using `joblib`

---

## 📁 Project Structure

```
heart-disease-predictor/
├── heart-disease-detect.ipynb     # Main Jupyter notebook
├── lgbm_pipeline.pkl              # Saved model pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This documentation file
```

---

## 🔧 Technologies Used

- Python
- LightGBM
- scikit-learn
- SHAP
- Pandas, NumPy, Matplotlib, Seaborn

---

## 🧪 Evaluation Metrics

- **Accuracy**: 100%
- **Confusion Matrix**:
  ```
  [[102   0]
   [  0 103]]
  ```
- **Top SHAP Features**:
  - `cp` (chest pain type)
  - `ca` (number of major vessels)
  - `sex`, `thal`, `oldpeak`

---

## 💡 How to Use

### 🔨 Installation

```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
```

### 🚀 Run the Notebook

```bash
jupyter notebook heart-disease-detect.ipynb
```

### 🧠 Predict with Saved Model

```python
import joblib
import pandas as pd

pipeline = joblib.load("lgbm_pipeline.pkl")
sample = pd.DataFrame([your_input_data], columns=[...])  # Use correct feature names
prediction = pipeline.predict(sample)
```

---

## 📷 SHAP Visuals

Includes:
- SHAP bar plot for feature importance
- Beeswarm plot for distribution & impact
- Force & waterfall plots for individual explanation

---

## ✍️ Author

**Nasir Hasan Ashik**  
📧 Email: your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)  
🔗 [GitHub](https://github.com/your-username)

---

## 🏁 License

This project is licensed under the MIT License.
