import shap
import pandas as pd
import numpy as np

def generate_shap_summary(model, data):
    """
    Generate SHAP summary with improved data handling
    """
    try:
        # Ensure data is in the right format
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input data must be pandas DataFrame or numpy array")
            
        # Convert to numpy if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
            
        # Create appropriate explainer
        if hasattr(model, "feature_importances_"):  # Tree-based model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_array)
            if isinstance(shap_values, list):  # Handle multi-class
                shap_values = shap_values[1]  # Use class 1
        else:  # Generic model
            explainer = shap.Explainer(model.predict, data_array)
            shap_values = explainer(data_array)
            
        # Calculate mean absolute SHAP values
        if len(shap_values.shape) == 3:  # For multi-class outputs
            shap_values = np.abs(shap_values).mean(0)
        else:
            shap_values = np.abs(shap_values)
            
        mean_shap = pd.Series(np.mean(shap_values, axis=0))
        
        # Get feature names
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns
        else:
            feature_names = [f"Feature {i}" for i in range(data.shape[1])]
            
        mean_shap.index = feature_names
        mean_shap = mean_shap.sort_values(ascending=False)
        
        # Format summary
        summary = "Top 10 important features:\n\n"
        for feature, value in mean_shap.head(10).items():
            summary += f"- {feature}: {value:.4f}\n"
            
        return summary
        
    except Exception as e:
        raise ValueError(f"SHAP calculation failed: {str(e)}")