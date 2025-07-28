import shap
import numpy as np

class ShapExplainer:
    def __init__(self, model, background_data = None):
        self.model = model
        self.explainer = None
        self.background_data = background_data

    def create_explainer(self):
        model_type_str = str(type(self.model)).lower()
        if 'tree' in model_type_str:
            self.explainer = shap.TreeExplainer(self.model, self.background_data)
        else:
            self.explainer = shap.Explainer(self.model.predict, self.background_data)
    
    def generate_shap_values(self, data):

        try:
            if self.explainer is None:
                self.create_explainer()

            if hasattr(data, 'values'):
                data_array = data.values
            else:
                data_array = np.array(data)
                
            data_array = data.values if hasattr(data, 'values') else np.array(data)

            if isinstance(self.explainer, shap.explainers._tree.TreeExplainer):
                shap_values = self.explainer.shap_values(data_array)
            else:
                explanation = self.explainer(data_array)
                shap_values = explanation.values

            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    return shap_values[1]
                else:
                    return np.mean(shap_values, axis=0)
            return shap_values
                
        except Exception as e:
            raise ValueError(f"Error calculating SHAP values: {str(e)}")
        
    @property
    def expected_value(self):
        if self.explainer is None:
            raise ValueError("Explainer not created yet.")
        ev = getattr(self.explainer, "expected_value", None)
        if isinstance(ev, (list, np.ndarray)):
            if len(ev) == 2:
                return ev[1]
            else:
                return np.mean(ev)
        return ev