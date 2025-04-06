import shap
import numpy as np

class ShapExplainer:
    def __init__(self, model):
        self.model = model
    
    def generate_shap_values(self, data):
        """Generate consistent SHAP values for different model types"""
        try:
            # Convert data to numpy array if needed
            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
                
            # Handle different model types
            if str(type(self.model)).lower().find('tree') != -1:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(data_array)
                
                # Handle classification models case (list of arrays)
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:  # Binary case
                        return shap_values[1]  # We take values for positive class
                    else:
                        return np.mean(shap_values, axis=0)  # Average for multiclass
                return shap_values
                
            else:  # For non-tree-based models
                explainer = shap.Explainer(self.model.predict, data_array)
                shap_values = explainer(data_array)
                return shap_values.values
                
        except Exception as e:
            raise ValueError(f"Error calculating SHAP values: {str(e)}")