class ShapPrompts:
    @staticmethod
    def get_analysis_prompt(model_name, shap_values, data_shape):
        """Genera el prompt para el LLM basado en los resultados SHAP"""
        feature_importance = "\n".join(
            f"- {i}: {v:.4f}" for i, v in enumerate(shap_values.mean(axis=0))
        )
        
        return f"""
You are a data science expert explaining model behavior. 

Model: {model_name}
Dataset Shape: {data_shape[0]} samples, {data_shape[1]} features

Feature Importance:
{feature_importance}

Please provide a concise explanation that:
1. Starts with a one-sentence plain English summary
2. Explains the top 3 features in business terms
3. Mentions any surprising patterns
4. Concludes with practical recommendations
5. Uses bullet points for readability

Write professionally but conversationally, avoiding technical jargon.
Keep the explanation under 200 words.
"""
