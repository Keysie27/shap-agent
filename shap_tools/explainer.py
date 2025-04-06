import shap
import numpy as np

class ShapExplainer:
    def __init__(self, model):
        self.model = model
    
    def generate_shap_values(self, data):
        """Genera valores SHAP y los convierte a array NumPy"""
        if hasattr(self.model, 'feature_importances_'):  # Para modelos basados en árboles
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(data)
            return np.array(shap_values[1] if isinstance(shap_values, list) else shap_values)
        else:  # Para otros modelos
            explainer = shap.Explainer(self.model.predict, data)
            return explainer(data).values  # Extraemos solo los valores