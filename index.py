import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv(r"C:\Users\hp\Downloads\mobile_price_dataset.csv")

df.head()
df.info()
print("Raw shape:", df.shape)

df = df.drop_duplicates()
print("After:", df.shape)

df.isnull().sum()
df["battery_mah"] = df["battery_mah"].fillna(df["battery_mah"].mean())
df["camera_mp"]   = df["camera_mp"].fillna(df["camera_mp"].mean())
df.info()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.figure(figsize=(10, 10))
sns.boxplot(x='brand', y='price_usd', data=df)

plt.figure(figsize=(10, 10))
sns.barplot(x='brand', y='price_usd', data=df)

sns.countplot(x="ram_gb", hue="price_range_label", data=df)

plt.figure(figsize=(10, 10))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5)
sns.countplot(x='has_5g', hue='price_range_label', data=df)


print(df['brand'].value_counts())
print(df['ram_gb'].value_counts())
print(df['storage_gb'].value_counts())
print(df['camera_mp'].value_counts())
print(df['screen_size_inch'].value_counts())
print(df['front_camera_mp'].value_counts())
print(df['has_5g'].value_counts())
print(df['processor_speed_ghz'].value_counts())
print(df['has_nfc'].value_counts())
print(df['cores'].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["os"] = le.fit_transform(df["os"])

df = pd.get_dummies(df, columns=['brand', 'price_range_label'], drop_first=True)

cols = ['brand_Infinix', 'brand_Motorola', 'brand_Nokia', 'brand_OnePlus',
        'brand_Oppo', 'brand_Realme', 'brand_Samsung', 'brand_Tecno',
        'brand_Vivo', 'brand_Xiaomi', 'price_range_label_Mid-Range',
        'price_range_label_Premium', 'price_range_label_Upper Mid-Range']

existing_cols = [c for c in cols if c in df.columns]
df[existing_cols] = df[existing_cols].astype(int)

# ── 4. Build X and y ───────────────────────────────────────────────────────────
price_range_cols = [c for c in df.columns if c.startswith('price_range_label_')]

X = df.drop(['price_usd'] + price_range_cols, axis=1)

# Save column order — required to align prediction inputs later
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.joblib')
print("Feature columns saved:", feature_columns)

y_text = df[price_range_cols].idxmax(axis=1)
le_y = LabelEncoder()
y_category = le_y.fit_transform(y_text)
y_price = df['price_usd']

# ── 5. Train / test split — FIX: test_size reduced from 0.42 → 0.20 ───────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_category_train, y_category_test, y_price_train, y_price_test = train_test_split(
    X, y_category, y_price, test_size=0.20, random_state=42)   # was 0.42

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 6. Classification ──────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, roc_auc_score)
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_category_train)
y_pred = lr_model.predict(X_test)
print("\n── Logistic Regression ──")
print("Accuracy:", accuracy_score(y_category_test, y_pred))
print("Precision:", precision_score(y_category_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_category_test, y_pred, average='weighted'))
print("F1:", f1_score(y_category_test, y_pred, average='weighted'))
print(classification_report(y_category_test, y_pred))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
rf_model.fit(X_train, y_category_train)
rf_pred = rf_model.predict(X_test)
print("\n── Random Forest ──")
print("Accuracy:", accuracy_score(y_category_test, rf_pred))
print("Precision:", precision_score(y_category_test, rf_pred, average='weighted'))
print("Recall:", recall_score(y_category_test, rf_pred, average='weighted'))
print("F1:", f1_score(y_category_test, rf_pred, average='weighted'))
print(classification_report(y_category_test, rf_pred))

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_category_train)
xgb_pred = xgb_model.predict(X_test)
print("\n── XGBoost ──")
print("Accuracy:", accuracy_score(y_category_test, xgb_pred))
print("Precision:", precision_score(y_category_test, xgb_pred, average='weighted'))
print("Recall:", recall_score(y_category_test, xgb_pred, average='weighted'))
print("F1:", f1_score(y_category_test, xgb_pred, average='weighted'))
print(classification_report(y_category_test, xgb_pred))

# ROC curve (multiclass one-vs-rest)
y_test_binarized = label_binarize(y_category_test, classes=[0, 1, 2])
y_score = xgb_model.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
for i, class_name in enumerate(le_y.classes_):
    RocCurveDisplay.from_predictions(
        y_test_binarized[:, i],
        y_score[:, i],
        name=f"ROC curve for {class_name}",
        ax=ax
    )
plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve (One-vs-Rest)")
plt.legend()
plt.show()

macro_auc = roc_auc_score(y_category_test, y_score, multi_class="ovr", average="macro")
print(f"Overall Macro ROC AUC Score: {macro_auc:.4f}")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Linear Regression
li_model = LinearRegression()
li_model.fit(X_train, y_price_train)
li_pred = li_model.predict(X_test)
print("\n── Linear Regression ──")
print("MAE:", mean_absolute_error(y_price_test, li_pred))
print("R²:", r2_score(y_price_test, li_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_price_test, li_pred)))
print("Average price:", df['price_usd'].mean())

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train, y_price_train)
print("\nRandom Forest Regressor MAE:", mean_absolute_error(y_price_test, rf_reg.predict(X_test)))

# XGBoost Regressor
xgb_reg = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="mae")
xgb_reg.fit(X_train, y_price_train)
xgb_price_pred = xgb_reg.predict(X_test)
print("\nXGBoost Price MAE:", mean_absolute_error(y_price_test, xgb_price_pred))
print("XGBoost R²:", r2_score(y_price_test, xgb_price_pred))

joblib.dump(xgb_model, 'phone_range_model.joblib')
joblib.dump(rf_reg,    'phone_price_model.joblib')
joblib.dump(le_y,      'target_encoder.joblib')
joblib.dump(le,        'os_encoder.joblib')        
print("Models and encoders saved!")
