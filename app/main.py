# Streamlit UI logic: file upload, SHAP explanation, agent output
import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import shap
from io import BytesIO
from app.file_handler import load_dataset
from utils.shap_explainer import generate_shap_summary
from agent.shap_agent import explain_with_agent, check_ollama_alive

# Page setup
st.set_page_config(page_title="SHAP-Agent", layout="wide")
st.title("🤖 SHAP-Agent: Model Explanation with SHAP + LLM")

# Check Ollama
if not check_ollama_alive():
    st.warning("⚠️ Ollama is not running. Please run `ollama run mistral` in your terminal.")

# Model options
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# UI Elements
selected_model_name = st.selectbox(
    "Select a model to analyze:",
    list(MODEL_OPTIONS.keys())
)

data_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if selected_model_name and data_file:
    # Model and data loading section
    with st.spinner("Loading model and dataset..."):
        try:
            # Load model
            model_path = os.path.join("models", "sample_models", MODEL_OPTIONS[selected_model_name])
            model = joblib.load(model_path)
            
            # Validate model
            if not hasattr(model, "predict"):
                st.error("❌ Invalid model: Must have predict() method")
                st.stop()
            
            # Load and validate dataset
            data = load_dataset(data_file)
            
            # Show preview
            st.write("📄 Dataset preview:", data.head())
            
            # Check data types
            if not all(pd.api.types.is_numeric_dtype(dt) for dt in data.dtypes):
                st.warning("⚠️ Non-numeric columns detected. Converting...")
                data = data.apply(pd.to_numeric, errors='coerce')
                if data.isnull().any().any():
                    st.error("❌ Could not convert all columns to numeric")
                    st.stop()
            
            # Remove target if exists
            if 'target' in data.columns:
                data = data.drop(columns=['target'])
            
            st.success(f"✅ Model loaded: {selected_model_name}")
            st.success(f"✅ Dataset loaded. Shape: {data.shape}")
            
        except Exception as e:
            st.error(f"❌ Loading error: {str(e)}")
            st.stop()

    # SHAP section
    shap_values = None
    with st.spinner("Generating SHAP summary..."):
        try:
            explainer = shap.Explainer(model, data)
            shap_values = explainer(data)
            
            if hasattr(shap_values, 'values'):
                abs_shap = np.abs(shap_values.values)
            else:
                abs_shap = np.abs(shap_values)
            
            mean_shap = pd.Series(np.mean(abs_shap, axis=0))
            mean_shap.index = data.columns
            mean_shap = mean_shap.sort_values(ascending=False)
            
            # Display SHAP values
            st.subheader("📊 SHAP Values")
            st.write("Raw SHAP values:", shap_values.values)
            st.write("Feature importance:", mean_shap)
            
            shap_summary = "Top 10 important features:\n\n"
            for feature, value in mean_shap.head(10).items():
                shap_summary += f"- {feature}: {value:.4f}\n"
                
            st.success("✅ SHAP analysis completed")
            
        except Exception as e:
            st.error(f"❌ SHAP error: {str(e)}")
            st.stop()

    # AI Explanation section
    explanation = None
    with st.spinner("Generating AI explanation..."):
        try:
            # Force English explanation by modifying the prompt
            english_prompt = f"""
            You are an AI assistant that explains machine learning models in English.
            This dataset has {data.shape[0]} rows and {data.shape[1]} features.

            Given this SHAP summary, explain in clear English what the model is focusing on and why:

            SHAP Summary:
            {shap_summary}

            Provide the explanation in English only, using simple terms suitable for non-experts.
            """
            
            explanation = explain_with_agent(english_prompt, data.shape)
            st.success("✅ Explanation generated")
        except Exception as e:
            st.error(f"❌ Explanation error: {str(e)}")
            st.stop()

    # Results section
    if explanation:
        st.subheader("🧠 Model Explanation")
        st.write(explanation)

        # Download button
        st.download_button(
            "📥 Download Explanation",
            explanation,
            file_name=f"{selected_model_name.lower().replace(' ', '_')}_explanation.txt",
            mime="text/plain"
        )

# Sidebar info
st.sidebar.markdown("""
### ℹ️ Instructions:
1. Select a pre-trained model
2. Upload your dataset (CSV format)
3. Wait for the explanation

### 📌 Notes:
- Models must be in `models/sample_models/`
- Dataset should contain only features (no target column)
- Non-numeric columns will be automatically converted
""")