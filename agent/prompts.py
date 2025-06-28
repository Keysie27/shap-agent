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
        You are a data science expert explaining machine learning model behavior to business stakeholders.

        Model: {model_name}  
        Dataset Shape: {data.shape[0]} samples, {data.shape[1]} features

        Top {top_n} Most Important Features:  
        {feature_info}

        **Please provide a model explanation using the following structure exactly as written. Use the section headers below verbatim (with double asterisks and number):**

        **1. Summary**  
        A one-sentence summary of the top 3 most influential features in the model. Example:  
        "The **{model_name}** model is primarily influenced by: **feature1**, **feature2**, and **feature3**."

        **2. Top Feature Analysis**  
        For each of the top 3 features:  
        - **Feature Name**:  
            • Direction of impact (positive/negative)  
            • Relative importance compared to others  
            • Potential business interpretation

        **3. Key Observations**  
        • Note any surprising relationships  
        • Highlight unexpected feature rankings  
        • Mention notable absences from top features

        **4. Practical Recommendations**  
        Provide 5 specific, actionable suggestions:  
        • Recommendation related to top features  
        • Suggestion for future data collection  
        • Tip for monitoring model performance  
        • Strategy suggestion for business users  
        • Opportunity for further analysis

        **Formatting Requirements:**  
        - Use bullet points (•) for lists  
        - Keep feature names in **bold**  
        - Use simple business language (avoid technical jargon)  
        - Keep the full response under 200 words  
        - Avoid markdown other than bold and bullets
        """