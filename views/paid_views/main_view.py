import re
import streamlit as st
from agent.shap_agent import ShapAgent
from services.helpers import get_img_base_64
from services.pdf_generator import create_shap_report_pdf
from shap_tools.explainer import ShapExplainer
from shap_tools.visualizations import ShapVisualizer
from app.file_handler import load_dataset
from agent.prompts import ShapPrompts
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pdf_bytes = None

def main_view():
    st.set_page_config(page_title="SHAP-Agent", layout="wide")

    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None

    #'''
    st.markdown("""
    <style>
    /* Hide the toolbar (hamburger + status) */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Optional: Also hide the header if it reappears */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* Reclaim vertical space */
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)#'''

    ##render each individual component

    # Sidebar toggle button
    _render_toggle_button()

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
            st.markdown('<span id="button-after2"></span>', unsafe_allow_html=True)
            analyze_button = st.button("✨ Analyze model with AI ✨", use_container_width=True)

        if analyze_button:
            _run_analysis(model_name, data_file)

# Helper Methods 

def _set_custom_css():
    with open("assets/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _render_toggle_button():
    st.markdown('<span id="button-after1"></span>', unsafe_allow_html=True)
    if st.button("☰"):
        st.session_state.sidebar_mode = (
        'About' if st.session_state.sidebar_mode == 'Instructions' else 'Instructions'
    )
    
def _render_download_button():
    pdf_data = st.session_state.get('pdf_bytes')
    if pdf_data:
        st.markdown('<span id="button-after3"></span>', unsafe_allow_html=True)
        st.download_button(
            label="📥",
            data=pdf_data,
            file_name="shap_report.pdf",
            mime="application/pdf",
            key="download_pdf_icon"
        )
    else:
        st.warning("PDF not available yet.")

def _render_header():
    st.title("🤖 SHAP-Agent: Model Explanation")
    st.markdown("**This tool helps you to understand how your machine learning model makes decisions!**")
    st.markdown("⚡ Powered by SHAP (SHapley Additive exPlanations)")

def _render_sidebar():
    # Initialize toggle state if not set
    if 'sidebar_mode' not in st.session_state:
        st.session_state.sidebar_mode = 'Instructions'

    # Sidebar content based on mode
    with st.sidebar:
        if st.session_state.sidebar_mode == 'Instructions':
            st.markdown("""
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
        else:
            st.markdown("""
            ### Historial
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
    shap_summary_img_base64 = None
    bar_chart_img_base64 = None

    try:
        with st.spinner("Loading model and data..."):
            try:
                model_path = os.path.join("models", "sample_models", model_name)
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded: {model_name}")
            except Exception as e:
                st.error("❌ Failed to load model.")
                raise e

            try:
                data = load_dataset(data_file)
                st.dataframe(data.head(), use_container_width=True)
                st.success(f"✅ Dataset loaded. Shape: {data.shape}")
            except Exception as e:
                st.error("❌ Failed to load dataset.")
                raise e

        # SHAP Analysis
        st.header("🔍 SHAP Analysis")
        tab_shap1, tab_shap2 = st.tabs(["📄 SHAP Values", "📊 Graph view"])
        try:
            explainer = ShapExplainer(model)
            visualizer = ShapVisualizer()
        except Exception as e:
            st.error("❌ Failed to initialize SHAP tools.")
            raise e

        with tab_shap1:
            try:
                with st.spinner("Calculating SHAP values..."):
                    shap_values = explainer.generate_shap_values(data)
                    shap_df = pd.DataFrame(
                        shap_values[0] if len(shap_values.shape) == 3 else shap_values,
                        columns=data.columns
                    )
                    st.dataframe(shap_df.head(), use_container_width=True)
            except Exception as e:
                st.error("❌ Failed to compute SHAP values.")
                raise e

        with tab_shap2:
            try:
                with st.spinner("Generating visualizations..."):
                    plots = visualizer.create_all_plots(shap_values, data)
                    summary_fig = plots.get('summary')
                    shap_summary_img_base64 = get_img_base_64(summary_fig)

                    for name in ['summary', 'bar', 'beeswarm']:
                        if name in plots:
                            st.markdown(f"### {name.capitalize()} Plot")
                            st.pyplot(plots[name])
                            plt.close(plots[name])
            except Exception as e:
                st.error("❌ Failed to generate SHAP plots.")
                raise e

        # Feature Importance
        st.header("📊 Feature Importance")
        tab1, tab2 = st.tabs(["📄 Feature details", "📊 Graph view"])
        try:
            with tab1:
                mean_shap = np.abs(shap_values).mean(axis=0)
                if len(mean_shap.shape) > 1:
                    mean_shap = mean_shap.mean(axis=0)
                importance_df = pd.DataFrame({
                    'Feature': data.columns,
                    'Importance': mean_shap
                }).sort_values('Importance', ascending=False).head(5)
                st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}).hide(axis='index'), use_container_width=True)
        except Exception as e:
            st.error("❌ Failed to compute feature importance.")
            raise e

        try:
            with tab2:
                if 'importance' in plots:
                    fig = plots['importance']
                    summary_fig = plots.get('importance')
                    bar_chart_img_base64 = get_img_base_64(summary_fig)           
                    fig.set_size_inches(10, 6)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
        except Exception as e:
            st.error("❌ Failed to render feature importance plot.")
            raise e

        # Model Insights
        st.header("🧠 Model Insights")
        try:
            agent = ShapAgent()
            with st.spinner("Generating explanation..."):
                prompt = ShapPrompts.get_analysis_prompt(model_name, shap_values, data)
                explanation = agent.generate_explanation(prompt, data.shape)
                st.markdown(explanation)
        except Exception as e:
            st.error("❌ Failed to generate AI explanation.")
            raise e

        # PDF Generation
        try:
            sections = re.split(r"\*\*\d+\.\s?", explanation.strip())
            cleaned_data = [section.replace('*', '').replace('\n', '') for section in sections]

            summary = cleaned_data[1].strip('"')
            top_feature_analysis = re.split(r"- (?=Feature Name)", cleaned_data[2].strip())[1:]
            top_feature_analysis = [item.strip() for item in top_feature_analysis]
            key_observations = [part.strip() for part in cleaned_data[3].replace('Key Observations', '').split('- ') if part.strip()]
            practical_recommendations = [part.strip() for part in cleaned_data[4].replace('Practical Recommendations', '').split('- ') if part.strip()]

            output_pdf_path = "output/shap_report.pdf"
            create_shap_report_pdf(
                output_pdf_path,
                shap_summary_img_base64=shap_summary_img_base64,
                bar_chart_img_base64=bar_chart_img_base64,
                top_influencers_sentence=summary,
                feature_analysis_points=top_feature_analysis,
                key_observations_points=key_observations,
                practical_recommendations=practical_recommendations
            )

            with open(output_pdf_path, "rb") as f:
                st.session_state.pdf_bytes = f.read()

            _render_download_button()

        except Exception as e:
            st.error("❌ Failed to generate PDF report.")
            raise e

    except Exception as e:
        st.error(f"❌ General Error: {str(e)}")
        st.error("""
        Troubleshooting:
        - Check dataset structure
        - Try with a smaller dataset
        """)
