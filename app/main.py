import streamlit as st
from agent.shap_agent import ShapAgent
from shap_tools.explainer import ShapExplainer
from shap_tools.visualizations import ShapVisualizer
from app.file_handler import load_dataset
import joblib
import os
import numpy as np
import pandas as pd
from agent.prompts import ShapPrompts
import matplotlib.pyplot as plt

def main():
    # Initial app configuration
    st.set_page_config(page_title="SHAP-Agent", layout="wide")
    st.title("🤖 SHAP-Agent: Model Explanation")
    st.markdown("""
    This tool helps you understand how your machine learning model makes decisions using SHAP (SHapley Additive exPlanations).
    Upload your dataset and select a model to get feature importance analysis and model behavior explanation.
    """)

    # Sidebar info
    st.sidebar.markdown("""
    ### ℹ️ Instructions:
    1. Select a pre-trained model.
    2. Upload your dataset. Supported formats: CSV.
    3. Wait for the explanation

    ### 📌 Notes:
    - Dataset should contain only features (no target column)
    - Non-numeric columns will be automatically converted

    ### 📊 Visualizations Guide:
    - **Feature Importance**: Shows which features matter most
    - **Impact Distribution**: Reveals how features affect predictions
    - **Dependence Plots**: Show relationships for top features
    """)

    # Check Ollama status
    with st.spinner("Checking Ollama service..."):
        if not ShapAgent.check_ollama_alive():
            st.warning("⚠️ Ollama is not running. Please run `ollama run mistral`")
            st.stop()

    # UI Elements
    st.subheader("Select a model to analyze:")
    model_name = st.selectbox(
        "Model:", 
        ["logistic_regression.pkl", "random_forest.pkl"],
        label_visibility="collapsed"
    )
    
    st.subheader("Upload your dataset:")
    data_file = st.file_uploader(
        "Choose a CSV file", 
        type=["csv"],
        label_visibility="collapsed"
    )

    if model_name and data_file:
        try:
            # Load resources
            with st.spinner("Loading model and data..."):
                model_path = os.path.join("models", "sample_models", model_name)
                model = joblib.load(model_path)
                data = load_dataset(data_file)
                
                st.success(f"✅ Model loaded: {model_name}")
                st.dataframe(data.head(), use_container_width=True)
                st.success(f"✅ Dataset loaded. Shape: {data.shape}")

            # SHAP Analysis
            st.header("🔍 SHAP Analysis")
            explainer = ShapExplainer(model)
            visualizer = ShapVisualizer()
            
            with st.spinner("Calculating SHAP values..."):
                shap_values = explainer.generate_shap_values(data)
                
                # Show raw SHAP values
                with st.expander("View Raw SHAP Values", expanded=True):
                    shap_df = pd.DataFrame(
                        shap_values[0] if len(shap_values.shape) == 3 else shap_values,
                        columns=data.columns
                    )
                    st.dataframe(shap_df.head(), use_container_width=True)

            # SHAP Visualizations (collapsed by default)
            with st.expander("📊 SHAP Visualizations", expanded=False):
                with st.spinner("Generating visualizations..."):
                    plots = visualizer.create_all_plots(shap_values, data)
                    
                    # Display summary plot
                    if 'summary' in plots:
                        st.pyplot(plots['summary'])
                        plt.close(plots['summary'])

            # Generate explanation
            st.header("🧠 Model Insights")
            agent = ShapAgent()
            
            with st.spinner("Generating explanation..."):
                prompt = ShapPrompts.get_analysis_prompt(
                    model_name=model_name,
                    shap_values=shap_values,
                    data=data
                )
                explanation = agent.generate_explanation(
                    summary_text=prompt,
                    data_shape=data.shape
                )
                st.markdown(explanation)

            # Feature importance (always visible)
            st.header("📊 Feature Importance")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                with st.spinner("Calculating feature importance..."):
                    mean_shap = np.abs(shap_values).mean(axis=0)
                    if len(mean_shap.shape) > 1:
                        mean_shap = mean_shap.mean(axis=0)
                    importance_df = pd.DataFrame({
                        'Feature': data.columns,
                        'Importance': mean_shap
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    st.dataframe(
                        importance_df.style.format({'Importance': '{:.4f}'}),
                        use_container_width=True
                    )
            
            with col2:
                # Use the importance plot from visualizer
                if 'importance' in plots:
                    st.pyplot(plots['importance'])
                    plt.close(plots['importance'])

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("""
            Troubleshooting:
            1. Ensure Ollama is running (`ollama serve`)
            2. Pull the model (`ollama pull mistral`)
            3. Check CSV file format
            4. Try with a smaller dataset
            """)

if __name__ == "__main__":
    main()