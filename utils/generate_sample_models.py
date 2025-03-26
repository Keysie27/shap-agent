import pandas as pd
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Get root directory (shap-agent/)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Output paths
models_dir = os.path.join(base_dir, "models", "sample_models")
data_dir = os.path.join(base_dir, "data", "sample_data")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Load and split
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Save sample datasets (features only)
X_train.to_csv(os.path.join(data_dir, "random_forest.csv"), index=False)
X_train.to_csv(os.path.join(data_dir, "logistic_regression.csv"), index=False)

# Train and save models
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))

lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression.pkl"))

print("✅ Models and datasets saved successfully.")
