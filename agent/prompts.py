import numpy as np
import pandas as pd

class ShapPrompts:
    @staticmethod
    def get_analysis_prompt(model_name, shap_values, data, top_n=5):
        """Generate the LLM prompt based on SHAP results with actual feature names"""
        # Calculate mean absolute SHAP importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # If multidimensional (multiclass classification), average across classes
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=0)
        
        # Create dataframe with actual feature names
        importance_df = pd.DataFrame({
            'Feature': data.columns,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Format feature information
        feature_info = "\n".join(
            f"- {row['Feature']}: {row['Importance']:.4f}" 
            for _, row in importance_df.iterrows()
        )
        
        return f"""
You are a data science expert explaining model behavior. 

Model: {model_name}
Dataset Shape: {data.shape[0]} samples, {data.shape[1]} features

Top {top_n} Most Important Features:
{feature_info}

Please provide a concise explanation that:
1. Starts with a one-sentence plain English summary
2. Explains the top features using their ACTUAL NAMES from the list above
3. Mentions any surprising patterns
4. Concludes with practical recommendations
5. Uses bullet points for readability

IMPORTANT: 
- Always refer to features by their exact names from the list above
- Never make up feature meanings - only describe their impact
- If unsure about a feature's meaning, say so explicitly

Write professionally but conversationally, avoiding technical jargon.
Keep the explanation under 200 words.
"""