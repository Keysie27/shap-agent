import re
import os
import importlib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect

from agent.shap_agent import ShapAgent
from services.helpers import clear_analysis_data, get_img_base_64
from services.pdf_generator import create_shap_report_pdf
from shap_tools.explainer import ShapExplainer
from shap_tools.visualizations import ShapVisualizer
from app.file_handler import load_dataset
from agent.prompts import ShapPrompts

# ----------------------------
# Model registry
# ----------------------------
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression",
    "KNN": "knn",
    "Decision Tree": "decision_tree",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm",
    "Linear Regression": "linear_regression"
}

def home_view():
    st.set_page_config(page_title="SHAP-Agent", layout="wide")
    
    # Initialize ALL session state variables using .setdefault()
    st.session_state.setdefault('paid', True)
    st.session_state.setdefault('pdf_bytes', None)
    st.session_state.setdefault('analysis_started', False)
    st.session_state.setdefault('model_trained', False)
    st.session_state.setdefault('model', None)
    st.session_state.setdefault('model_name', "")
    st.session_state.setdefault('data', None)
    st.session_state.setdefault('target', None)
    st.session_state.setdefault('shap_values', None)
    st.session_state.setdefault('plots', {})
    st.session_state.setdefault('explanation', "")
    st.session_state.setdefault('shap_summary_img_base64', "")
    st.session_state.setdefault('page', "home")  # Add this if using page navigation

    _set_custom_css()
    _render_toggle_button()
    _render_header()
    _render_sidebar()
    _check_ollama()
    _model_and_data_selection()

    # Safe check using .get() for all session state access
    if st.session_state.get('model_trained', False) and st.session_state.get('model') is not None:
        _render_content()

# ----------------------------
# UI Components
# ----------------------------

def _set_custom_css():
    with open("shap-agent/assets/styles/styles.css") as f:
        css = f"""
        <style>
            {f.read()}
            /* Custom styles for buttons */
            .stButton>button {{
                width: 100%;
                transition: all 0.3s ease;
            }}
            .stButton>button:hover {{
                transform: scale(1.05);
            }}
            .analysis-button {{
                font-size: 16px !important;
                padding: 10px 15px !important;
            }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

def _render_toggle_button():
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.button("💎 Get Premium", key="toggle_button"):
        st.session_state.page = "payment"
        st.rerun()

def _render_download_button_disabled():
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    st.button("📥", help="PDF not available", disabled=True)

def _render_download_button_enabled():
    if st.session_state.get('pdf_bytes'):
        st.download_button(
            "📥 Download PDF Report", 
            data=st.session_state.pdf_bytes, 
            file_name="shap_report.pdf", 
            mime="application/pdf",
            help="Click to download the full SHAP analysis report",
            key="pdf_download"
        )

def _render_header():
    st.title("🤖 SHAP-Agent: Model Explanation")
    st.markdown("**Understand how your machine learning model makes decisions!**")
    st.markdown("⚡ Powered by SHAP (SHapley Additive exPlanations)")

def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        ### ℹ️ Instructions:
        1. Select a model
        2. Upload a dataset with target
        3. Train and analyze!
        ---
        ### Dataset Notes:
        - Include your target column.
        - Non-numeric features are auto-converted if possible.
        """)

def _check_ollama():
    with st.spinner("Checking Ollama service..."):
        if not ShapAgent.check_ollama_alive():
            st.error("⚠️ Ollama is not running. Please run `ollama run mistral`")
            st.stop()

# ----------------------------
# Model + Data Selection
# ----------------------------

def _model_and_data_selection():
    st.subheader("1. Select a model to train:")

    selected_display_name = st.selectbox("Choose a model:", list(MODEL_REGISTRY.keys()), key="model_select")
    module_name = MODEL_REGISTRY[selected_display_name]

    model_params = {}
    if module_name == "knn":
        model_params["n_neighbors"] = st.number_input("Number of Neighbors (K)", min_value=1, max_value=100, value=3, step=1, key="knn_neighbors")
    elif module_name == "decision_tree":
        model_params["max_depth"] = st.number_input("Max Tree Depth", min_value=1, max_value=100, value=5, step=1, key="tree_depth")

    st.subheader("2. Upload your dataset (with target):")
    data_file = st.file_uploader("Upload CSV", type=["csv"], key="data_uploader")

    if data_file:
        data = pd.read_csv(data_file)
        st.dataframe(data.head(3))
        target_column = st.selectbox("Select your target column:", data.columns, key="target_select")

        if target_column and st.button("✨ Train model ✨", key="train_button", type="primary"):
            with st.spinner("🚀 Training model. Please wait..."):
                try:
                    X = data.drop(columns=[target_column])
                    y = data[target_column]

                    model_module = importlib.import_module(f"models.sample_models.{module_name}")
                    model = model_module.train(X, y, **model_params)

                    st.session_state.update({
                        'model': model,
                        'model_name': selected_display_name,
                        'data': X,
                        'target': y,
                        'analysis_started': True,
                        'model_trained': True,
                        'training_message': f"✅ {selected_display_name} trained successfully!"
                    })

                    clear_analysis_data()
                    st.balloons()

                    st.markdown("---")
                    # ✅ Renderizar directamente la sección de análisis
                    _render_content()

                except Exception as e:
                    st.session_state.model_trained = False
                    st.error(f"❌ Training failed: {str(e)}")

def _render_content():
    # Mostramos el mensaje persistente si existe
    if 'training_message' in st.session_state:
        st.success(st.session_state.training_message)

    # Opciones de análisis
    st.header("🔍 Analysis Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Run SHAP Analysis", key="shap_btn"):
            _render_shap_analysis()
    
    with col2:
        if st.button("📈 Feature Impact", key="feature_btn"):
            _render_feature_impact()
    
    with col3:
        if st.button("🧠 AI Explanation", key="explain_btn"):
            _render_explanation_generation()
    
    st.markdown("---")
    st.header("📄 Report Generation")
    _render_pdf_generation()

# ----------------------------
# Main Content Sections
# ----------------------------


def _render_shap_analysis():
    if 'shap_values' not in st.session_state:
        with st.spinner("🔍 Calculating SHAP values. This may take a while..."):
            try:
                explainer = ShapExplainer(st.session_state.model)
                shap_values = explainer.generate_shap_values(st.session_state.data)
                st.session_state.shap_values = shap_values

                visualizer = ShapVisualizer()
                st.session_state.plots = visualizer.create_all_plots(shap_values, st.session_state.data)
                st.session_state.shap_summary_img_base64 = get_img_base_64(st.session_state.plots['summary'])
                
                st.success("✅ SHAP analysis completed!")
            except Exception as e:
                st.error(f"❌ SHAP calculation failed: {str(e)}")
                return

    st.header("📊 SHAP Results")
    tab1, tab2 = st.tabs(["Raw Values", "Visualizations"])

    with tab1:
        st.dataframe(
            pd.DataFrame(
                st.session_state.shap_values[0] if len(st.session_state.shap_values.shape) == 3 else st.session_state.shap_values,
                columns=st.session_state.data.columns
            ).head(),
            use_container_width=True
        )

    with tab2:
        if 'summary' in st.session_state.plots:
            st.subheader("Summary Plot")
            st.pyplot(st.session_state.plots['summary'])
        
        if 'bar' in st.session_state.plots:
            st.subheader("Feature Importance")
            st.pyplot(st.session_state.plots['bar'])
        
        plt.close('all')

def _render_feature_impact():
    if 'shap_values' not in st.session_state:
        st.warning("Please run SHAP analysis first")
        return

    st.header("📌 Top Features")
    
    mean_shap = np.abs(st.session_state.shap_values).mean(axis=0)
    if len(mean_shap.shape) > 1:
        mean_shap = mean_shap.mean(axis=0)

    impact_df = pd.DataFrame({
        'Feature': st.session_state.data.columns,
        'Impact': mean_shap
    }).sort_values('Impact', ascending=False).head(5)

    tab1, tab2 = st.tabs(["Table", "Chart"])
    
    with tab1:
        st.dataframe(
            impact_df.style.format({'Impact': '{:.4f}'}),
            use_container_width=True
        )
    
    with tab2:
        if 'importance' in st.session_state.plots:
            fig, ax = plt.subplots()
            impact_df.plot.barh(x='Feature', y='Impact', ax=ax)
            ax.set_title("Top 5 Most Important Features")
            st.pyplot(fig)
            plt.close(fig)

def _render_explanation_generation():
    if 'shap_values' not in st.session_state:
        st.warning("Please run SHAP analysis first")
        return

    with st.spinner("🤖 Generating AI-powered explanation..."):
        try:
            agent = ShapAgent()
            prompt = ShapPrompts.get_analysis_prompt(
                st.session_state.model_name,
                st.session_state.shap_values,
                st.session_state.data
            )
            explanation = agent.generate_explanation(prompt, st.session_state.data.shape)
            st.session_state.explanation = explanation
            
            st.header("🧠 Model Insights")
            st.markdown(explanation)
        except Exception as e:
            st.error(f"❌ Failed to generate explanation: {str(e)}")

def _render_pdf_generation():
    if 'explanation' not in st.session_state:
        st.warning("Generate model explanation first")
        return

    if st.button("🖨️ Generate PDF Report", key="pdf_gen_btn"):
        with st.spinner("Preparing PDF report..."):
            try:
                explanation = st.session_state.explanation
                plots = st.session_state.plots

                sections = re.split(r"\*\*\d+\.\s?", explanation.strip())
                cleaned_data = [section.replace('*', '').replace('\n', '').strip() for section in sections]

                summary = cleaned_data[1].strip('"') if len(cleaned_data) >= 2 else "No summary available."
                top_feature_analysis = [item.strip() for item in re.split(r"- (?=Feature Name|\*\*|[\w])", cleaned_data[2].strip()) if item.strip()] if len(cleaned_data) >= 3 else ["No feature analysis."]
                key_observations = [part.strip() for part in cleaned_data[3].split('- ') if part.strip()] if len(cleaned_data) >= 4 else ["No key observations."]
                practical_recommendations = [part.strip() for part in cleaned_data[4].split('- ') if part.strip()] if len(cleaned_data) >= 5 else ["No practical recommendations."]

                st.session_state.pdf_bytes = create_shap_report_pdf(
                    shap_summary_img_base64=st.session_state.shap_summary_img_base64,
                    bar_chart_img_base64=get_img_base_64(plots.get("importance")) if plots.get("importance") else None,
                    top_influencers_sentence=summary,
                    feature_analysis_points=top_feature_analysis,
                    key_observations_points=key_observations,
                    practical_recommendations=practical_recommendations
                )

                _render_download_button_enabled()
                st.success("✅ PDF report ready!")
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")