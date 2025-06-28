import streamlit as st
from agent.shap_agent import ShapAgent
from shap_tools.explainer import ShapExplainer
from shap_tools.visualizations import ShapVisualizer
from app.file_handler import load_dataset
from agent.prompts import ShapPrompts
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main_view():
    st.set_page_config(page_title="SHAP-Agent", layout="wide")

    ##render each individual component
    
    # Sidebar toggle button    
    _set_custom_css()

    _render_header()

    _render_sidebar()

    _check_ollama()

    # Model selection and data upload
    model_name, data_file = _model_and_data_selection()

    # Analysis button & processing
    if model_name and data_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("✨ Analyze model with AI ✨", use_container_width=True)

        if analyze_button:
            _run_analysis(model_name, data_file)

#  Helper functions 

def _set_custom_css():
    st.markdown("""<style>
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #9c27b0, #673ab7);
            color: white; border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s ease; font-weight: bold;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #7b1fa2, #5e35b1);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>""", unsafe_allow_html=True)

def _render_header():
    st.title("🤖 SHAP-Agent: Model Explanation")
    st.markdown("**This tool helps you to understand how your machine learning model makes decisions!**")
    st.markdown("⚡ Powered by SHAP (SHapley Additive exPlanations)")

def _render_sidebar():
    st.sidebar.markdown("""
    ### ℹ️ Instructions:
    1. Select a pre-trained model
    2. Upload your dataset
    3. Click the "Analyze" button
    ---
    ### 📋 Dataset Requirements:
    - Dataset should contain only features (no target column)
    - Non-numeric columns auto-converted
    ---
    ### 📊 Visualizations Guide:
    - **Feature Importance**: Top features
    - **Impact Distribution**: Feature effects
    """)

def _check_ollama():
    with st.spinner("Checking Ollama service..."):
        if not ShapAgent.check_ollama_alive():
            st.warning("⚠️ Ollama is not running. Please run `ollama run mistral`")
            st.stop()

def _model_and_data_selection():
    st.subheader("1. Select a model to analyze:")
    model_options = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl"
    }
    selected_display_name = st.selectbox("Model:", options=list(model_options.keys()), label_visibility="collapsed")
    model_name = model_options[selected_display_name]

    st.subheader("2. Upload your data:")
    data_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    return model_name, data_file

def _run_analysis(model_name, data_file):
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
        tab_shap1, tab_shap2 = st.tabs(["📄 SHAP Values", "📊 Graph view"])
        explainer = ShapExplainer(model)
        visualizer = ShapVisualizer()

        with tab_shap1:
            with st.spinner("Calculating SHAP values..."):
                shap_values = explainer.generate_shap_values(data)
                shap_df = pd.DataFrame(
                    shap_values[0] if len(shap_values.shape) == 3 else shap_values,
                    columns=data.columns
                )
                st.dataframe(shap_df.head(), use_container_width=True)

        with tab_shap2:
            with st.spinner("Generating visualizations..."):
                plots = visualizer.create_all_plots(shap_values, data)
                for name in ['summary', 'bar', 'beeswarm']:
                    if name in plots:
                        st.markdown(f"### {name.capitalize()} Plot")
                        st.pyplot(plots[name])
                        plt.close(plots[name])

        # Feature importance table and graph
        st.header("📊 Feature Importance")
        tab1, tab2 = st.tabs(["📄 Feature details", "📊 Graph view"])
        with tab1:
            mean_shap = np.abs(shap_values).mean(axis=0)
            if len(mean_shap.shape) > 1:
                mean_shap = mean_shap.mean(axis=0)
            importance_df = pd.DataFrame({
                'Feature': data.columns,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=False).head(5)
            print(importance_df)
            print("a")
            st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}).hide(axis='index'), use_container_width=True)

        with tab2:
            if 'importance' in plots:
                fig = plots['importance']
                fig.set_size_inches(10, 6)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Model Insights
        st.header("🧠 Model Insights")
        agent = ShapAgent()
        with st.spinner("Generating explanation..."):
            prompt = ShapPrompts.get_analysis_prompt(model_name, shap_values, data)
            explanation = agent.generate_explanation(prompt, data.shape)
            st.markdown(explanation)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("""
        Troubleshooting:
        1. Ensure Ollama is running (`ollama serve`)
        2. Pull the model (`ollama pull mistral`)
        3. Check CSV file format
        4. Try with a smaller dataset
        """)

