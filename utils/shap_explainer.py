import shap
import pandas as pd

def generate_shap_summary(model, data):
    # Validate model
    if not hasattr(model, "predict"):
        raise ValueError("❌ The uploaded model is not valid. It must implement .predict().")

    # Drop target if present
    if "target" in data.columns:
        data = data.drop(columns=["target"])

    try:
        # Use TreeExplainer if it's a tree-based model
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)
            # If multi-output, use class 1
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
        else:
            # Generic fallback
            explainer = shap.Explainer(model, data)
            shap_values = explainer(data)

    except Exception as e:
        raise ValueError(f"❌ SHAP failed to explain the model:\n{e}")

    # Compute mean abs SHAP values
    shap_df = pd.DataFrame(shap_values, columns=data.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    summary_text = "Top 10 features influencing the model (SHAP Global Importance):\n\n"
    for feature, value in mean_abs.head(10).items():
        summary_text += f"- {feature}: {value:.4f}\n"

    return summary_text