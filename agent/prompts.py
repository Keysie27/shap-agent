import numpy as np
import pandas as pd

class ShapPrompts:
    @staticmethod
    def get_analysis_prompt(model_name, shap_values, data, top_n=5):
        mean_shap = np.abs(shap_values).mean(axis=0)
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=0)

        importance_df = pd.DataFrame({
            'Feature': data.columns,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False).head(top_n)

        feature_info = "\n".join(
            f"- {row['Feature']} ({row['Importance']:.4f})"
            for _, row in importance_df.iterrows()
        )

        return f"""
    You are a data analyst assistant. Explain the results of a machine learning model to business users using plain, non-technical language.

    Model used: {model_name}  
    Dataset shape: {data.shape[0]} rows, {data.shape[1]} features  
    Top {top_n} important features (by SHAP impact):  
    {feature_info}

    Follow this structure in your answer:

    **1. Summary**  
    One sentence stating the top 3 features and how they affect the model.

    **2. Feature Breakdown**  
    For each top feature:
    - Mention its role in predictions
    - Explain what an increase or decrease might imply

    **3. Observations**  
    - Any surprises in feature importance
    - Any missing or expected features

    **4. Actionable Tips**  
    Give 3 specific suggestions a business user can take based on the model.

    ⚠️ Avoid technical terms like "SHAP", "variance", or "coefficients".  
    ✅ Keep it under 200 words. Use bullet points where possible.
    """

    @staticmethod
    def get_advanced_analysis_prompt(model_name, shap_values, data, top_n=10):
        """Prompt for premium users: deeper explanation with technical and business insights"""
        mean_shap = np.abs(shap_values).mean(axis=0)

        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=0)

        importance_df = pd.DataFrame({
            'Feature': data.columns,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False).head(top_n)

        feature_info = "\n".join(
            f"- **{row['Feature']}** (SHAP importance: {row['Importance']:.4f})" 
            for _, row in importance_df.iterrows()
        )

        return f"""
        You are a senior data scientist explaining model behavior and technical insights to a team of product managers, analysts, and engineers.

        Model Type: **{model_name}**  
        Dataset Size: **{data.shape[0]} rows × {data.shape[1]} features**  
        Top {top_n} Features by SHAP Importance:  
        {feature_info}

        **Please follow this structure:**

        **1. Executive Summary**  
        Brief sentence on which features drive predictions and whether results are aligned with expectations.

        **2. Feature Interpretations**  
        For the top 3–5 features:
        - What behavior do they capture?
        - Are they positively or negatively associated?
        - Any interactions or conditional effects?

        **3. Unexpected Patterns**  
        - Any surprising features?
        - Missing expected signals?
        - Nonlinear or unstable relationships?

        **4. Recommendations**  
        - 2 ideas to improve data quality or add features
        - 2 ideas to use this model in business operations
        - 1 idea for continuous monitoring or retraining

        🧠 Use semi-technical language but remain clear.  
        ✍️ Prefer bullet points or concise blocks.  
        🎯 Stay under 350 words. No markdown beyond bold and numbered sections.
        """