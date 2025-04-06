import joblib
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(base_dir, "models", "sample_models")

# Lista de modelos a inspeccionar
model_files = ["random_forest.pkl", "logistic_regression_2.pkl"]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"\n🔍 Inspecting: {model_file}")
    
    model = joblib.load(model_path)
    
    print("✅ Model loaded successfully!")
    print(f"🧠 Model type: {type(model).__name__}")
    
    print("🔧 Model parameters:")
    for key, val in model.get_params().items():
        print(f" - {key}: {val}")
    
    # Opcional: mostrar feature_importances o coef_
    if hasattr(model, "feature_importances_"):
        print("📊 Feature importances:", model.feature_importances_)
    
    if hasattr(model, "coef_"):
        print("📉 Coefficients:", model.coef_)
    
    if hasattr(model, "classes_"):
        print("🏷️ Classes:", model.classes_)
