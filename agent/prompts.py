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

**Please provide a model explanation following this exact structure:**

**1. One-Sentence Summary**  
Start with: "The **{model_name}** model is primarily influenced by: [top 3 features]."

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
    • [Recommendation related to top feature]
    • [Recommendation about data collection]
    • [Recommendation about model monitoring]
    • [Recommendation about business process]
    • [Recommendation about further analysis]

**Formatting Requirements:**
- Always use bullet points (•) for lists
- Put feature names in **bold**
- Keep entire response under 200 words
- Use simple business language (no technical jargon)
- If feature meaning is unclear, state "The exact business meaning of [feature] requires domain knowledge
"""
