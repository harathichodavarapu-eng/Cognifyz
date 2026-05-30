# ===============================================================
# COGNIFYZ LEVEL 1 PROJECT
# PREDICT RESTAURANT RATINGS - FINAL CORRECT CODE
# Output Example:
# RMSE : 0.28
# R2 Score : 0.91
# Model Saved Successfully
# ===============================================================

# Install:
# pip install pandas numpy scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================================================
# LOAD DATASET
# ===============================================================

df = pd.read_csv(r"C:\Users\harat\OneDrive\Desktop\cognifyz\restaurant_data.csv")

print("Dataset Loaded Successfully")

# ===============================================================
# HANDLE MISSING VALUES
# ===============================================================

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna("Unknown")

# ===============================================================
# REMOVE UNUSED COLUMN
# ===============================================================

if "Restaurant ID" in df.columns:
    df.drop("Restaurant ID", axis=1, inplace=True)

# ===============================================================
# CONVERT ALL TEXT COLUMNS TO NUMBERS
# ===============================================================

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = pd.factorize(df[col])[0]

# ===============================================================
# FEATURES AND TARGET
# ===============================================================

target_column = "Aggregate rating"

X = df.drop(target_column, axis=1)
y = df[target_column]

# ===============================================================
# TRAIN TEST SPLIT
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ===============================================================
# MODEL TRAINING
# ===============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================================================
# PREDICTION
# ===============================================================

y_pred = model.predict(X_test)

# ===============================================================
# EVALUATION
# ===============================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE :", round(rmse, 2))
print("R2 Score :", round(r2, 2))

# ===============================================================
# FEATURE IMPORTANCE GRAPH
# ===============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ===============================================================
# SAVE MODEL
# ===============================================================

joblib.dump(model, "restaurant_rating_model.pkl")

print("Model Saved Successfully")