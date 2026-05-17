#  SmartPhone Price Predictor

A dual-model machine learning web app that predicts smartphone **price category** and **estimated price in USD** based on hardware specifications.

Built with Python, Scikit-learn, XGBoost, and Streamlit — deployed as an interactive web application.

---

##  What It Does

User enters phone specifications (RAM, storage, camera, battery, 5G, brand etc.) and the app returns:
- **Price Category** — Budget / Mid-Range / Upper Mid-Range / Premium
- **Confidence Score** — how confident the model is for each category
- **Estimated Price in USD** — predicted by a separate regression model

---

##  Models Used

| Task | Model | Metric |
|------|-------|--------|
| Price Category (Classification) | XGBoost Classifier | Accuracy, F1-Score, ROC-AUC |
| Exact Price (Regression) | Random Forest Regressor | MAE: ~$165 on avg price $548 |

**Model Selection Process:**
- Compared 3 classifiers: Logistic Regression, Random Forest, XGBoost
- Compared 3 regressors: Linear Regression, Random Forest, XGBoost
- Selected best model per task based on evaluation metrics

> **Note on Classification Accuracy:** All classifiers achieved 100% on this dataset because the data is synthetic with clearly defined price boundaries per category. In real-world data, brand perception and market positioning cause these boundaries to overlap, making classification genuinely harder. The regression model (MAE ~$165) reflects more realistic learning on continuous price prediction.

---

##  Tech Stack

- **Language:** Python
- **ML Models:** Scikit-learn, XGBoost
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit, Joblib

---

##  Dataset

- 1,500 rows, 16 columns
- Features: brand, RAM, storage, battery, camera, screen size, 5G, NFC, processor speed, cores, OS, year
- Target 1: price category (Budget / Mid-Range / Upper Mid-Range / Premium)
- Target 2: exact price in USD

**Data Cleaning Steps Performed:**
- Removed 15 duplicate rows
- Filled null values in `battery_mah` and `camera_mp` with column mean
- Encoded OS using LabelEncoder
- Encoded brand using get_dummies (one-hot encoding)

---

##  Evaluation Metrics

**Classification (XGBoost):**
- Accuracy, Precision, Recall, F1-Score (weighted)
- ROC-AUC Score (multiclass One-vs-Rest)
- ROC Curve plotted per class

**Regression (Random Forest):**
- MAE (Mean Absolute Error): ~$165
- R² Score
- RMSE (Root Mean Squared Error)

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/khadijaa-ali/SmartPhone-Price-Predictor.git
cd SmartPhone-Price-Predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train and save models**
```bash
python mobile_price_ml.py
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

---

##  Project Structure

```
SmartPhone-Price-Predictor/
│
├── mobile_price_ml.py        # Data cleaning, EDA, model training, saving
├── app.py                    # Streamlit web app
├── mobile_price_dataset.csv  # Dataset
├── phone_range_model.joblib  # Saved XGBoost classifier
├── phone_price_model.joblib  # Saved Random Forest regressor
├── target_encoder.joblib     # LabelEncoder for target
├── os_encoder.joblib         # LabelEncoder for OS
├── feature_columns.joblib    # Saved feature column order
└── requirements.txt          # Dependencies
```

---

## Key Learnings

- How to handle a dual-target ML problem (classification + regression on same dataset)
- Model comparison and selection based on multiple evaluation metrics
- Importance of feature encoding for categorical variables (brand, OS)
- How synthetic data differs from real-world data in model behavior
- End-to-end deployment from trained model to interactive Streamlit app

---

## 🔗 Connect

- **LinkedIn:** [Khadija Ali](https://www.linkedin.com/in/khadija-ali-5713a7325/)
- **Email:** dijaa.aali@gmail.com
