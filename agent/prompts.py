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
        """Prompt for premium users: deeper explanation with technical and business insights and sample cases"""
        import numpy as np
        import pandas as pd

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

        # Generate synthetic cases
        sample_cases = ""
        for i in range(min(2, len(data))):  # max 2 examples
            row = data.iloc[i]
            row_info = ", ".join(
                f"{col}={row[col]}" for col in data.columns[:5]  # show only first 5 columns
            )
            sample_cases += f"- For a user with {row_info}, the model might predict a similar outcome due to feature weights.\n"

        return f"""
    You are a senior data scientist explaining model behavior and technical insights to a team of product managers, analysts, and engineers.

    Model Type: **{model_name}**  
    Dataset Size: **{data.shape[0]} rows × {data.shape[1]} features**  
    Top {top_n} Features by SHAP Importance:  
    {feature_info}

    **Please follow this structure:**

    **1. Executive Summary**  
    Summarize which features most influence the predictions. Clarify whether these results align with domain expectations and any surprises.

    **2. Feature Interpretations**  
    For the top 3–5 features:
    - What user behavior or pattern does each represent?
    - Are they positively or negatively correlated with the output?
    - Do they show threshold effects or non-linear behavior?
    - Are there clear interactions with other features?

    **3. Unexpected Patterns**  
    - Are there features with unexpectedly high or low influence?
    - Are any expected drivers missing?
    - Mention unstable, volatile or counterintuitive effects.

    **4. Recommendations**  
    - Suggest 2 improvements to input data or data engineering (e.g., add features, fix skew).
    - Propose 2 business use cases for this model (e.g., early alerts, prioritization).
    - Describe 1 suggestion for model monitoring, drift detection or retraining frequency.

    **5. Example-Based Interpretability**  
    Provide up to 2 brief examples from the dataset showing how the model interprets real users:
    {sample_cases}

    🧠 Use semi-technical language but remain clear and structured.  
    ✍️ Use bullet points or short blocks.  
    🎯 Stay under 400 words. Use numbers and illustrative examples.
    Avoid using markdown except for section titles and bold text.
    """
