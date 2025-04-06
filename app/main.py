# Streamlit UI logic: file upload, SHAP explanation, agent output
import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from app.file_handler import load_dataset
from agent.shap_agent import explain_with_agent, check_ollama_alive

# Configure matplotlib style (updated to use valid style)
plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'

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

    # SHAP Visualization and Explanation Section
    with st.spinner("Generating SHAP insights..."):
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, data)
            shap_values = explainer(data)
            
            # Calculate feature importance
            if hasattr(shap_values, 'values'):
                abs_shap = np.abs(shap_values.values)
                raw_shap = shap_values.values
            else:
                abs_shap = np.abs(shap_values)
                raw_shap = shap_values
            
            mean_shap = pd.Series(np.mean(abs_shap, axis=0))
            mean_shap.index = data.columns
            mean_shap = mean_shap.sort_values(ascending=False)
            
            # ------------------------------------------
            # Enhanced SHAP Visualizations
            # ------------------------------------------
            st.subheader("🔍 SHAP Analysis")
            
            # Set larger figure size for all plots
            plt.rcParams['figure.figsize'] = [10, 6]
            
            # 1. Summary Plot
            st.write("### Feature Importance Overview")
            fig_summary, ax = plt.subplots()
            shap.summary_plot(shap_values, data, plot_type="bar", show=False)
            plt.tight_layout()  # Prevent label cutoff
            st.pyplot(fig_summary)
            plt.close()
            
            # 2. Beeswarm Plot (for detailed patterns)
            st.write("### Feature Impact Distribution")
            fig_beeswarm, ax = plt.subplots()
            shap.summary_plot(shap_values, data, show=False)
            plt.tight_layout()
            st.pyplot(fig_beeswarm)
            plt.close()
            
            # 3. Top Features Detailed View
            st.write("### Top 3 Influential Features")
            for feature in mean_shap.index[:3]:
                fig, ax = plt.subplots()
                shap.dependence_plot(feature, shap_values.values, data, show=False, ax=ax)
                plt.title(f"Dependence plot for {feature}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Display numerical summary
            st.write("### Feature Importance Scores")
            st.dataframe(mean_shap.reset_index().rename(
                columns={'index':'Feature', 0:'Importance'}).head(10).style.format({'Importance': '{:.4f}'}))
            
            shap_summary = "Top 10 important features:\n\n"
            for feature, value in mean_shap.head(10).items():
                shap_summary += f"- {feature}: {value:.4f}\n"
            
            # ------------------------------------------
            # Enhanced Explanation Prompt
            # ------------------------------------------
            enhanced_prompt = f"""
            You are a data science educator explaining machine learning models to business stakeholders.
            
            Dataset Overview:
            - Samples: {data.shape[0]}
            - Features: {data.shape[1]}
            - Model Type: {selected_model_name}
            - Top 5 Features: {', '.join(mean_shap.index[:5])}
            
            SHAP Analysis Results:
            {shap_summary}
            
            Please provide a concise explanation that:
            1. Starts with a one-sentence plain English summary
            2. Explains the top 3 features in business terms
            3. Mentions any surprising patterns
            4. Concludes with practical recommendations
            5. Uses bullet points for readability
            
            Write professionally but conversationally, avoiding technical jargon.
            Keep the explanation under 200 words.
            """
            
            # Generate and display explanation
            with st.spinner("Generating insightful explanation..."):
                explanation = explain_with_agent(enhanced_prompt, data.shape)
                
                st.subheader("🧠 Model Insights")
                st.markdown(explanation)
                
                st.download_button(
                    "📥 Download Full Report",
                    f"SHAP Analysis Report\n\nModel: {selected_model_name}\n\n{explanation}\n\n{shap_summary}",
                    file_name=f"{selected_model_name}_analysis_report.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"❌ Error in SHAP analysis: {str(e)}")
            st.stop()

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

### 📊 Visualizations Guide:
- **Feature Importance**: Shows which features matter most
- **Impact Distribution**: Reveals how features affect predictions
- **Dependence Plots**: Show relationships for top features
""")