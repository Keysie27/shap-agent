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

pdf_bytes = None

def home_view():
    st.set_page_config(page_title="SHAP-Agent", layout="wide")
    st.success(st.session_state.paid)

    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None

    _set_custom_css()
    _render_toggle_button()
    _render_download_button_disabled()
    _render_header()
    _render_sidebar()
    _check_ollama()

    model_name, X, y = _model_and_data_selection()

    if 'analysis_started' in st.session_state:
        _render_content(model_name, X, y)

# ---------------------------------------
# COMPONENTS
# ---------------------------------------

def _set_custom_css():
    with open("shap-agent/assets/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _render_toggle_button():
    st.markdown('<span id="button-after1"></span>', unsafe_allow_html=True)
    if st.button("💎"):
        st.session_state.page = "payment"
        st.rerun()

def _render_download_button_disabled():
    st.markdown('<span id="button-after4"></span>', unsafe_allow_html=True)
    if st.button("📥", help="PDF not available"):
        print("not ready yet")

def _render_download_button_enabled():
    pdf_data = st.session_state.get('pdf_bytes')
    if pdf_data:
        st.markdown('<span id="button-after3"></span>', unsafe_allow_html=True)
        st.download_button("📥", data=pdf_data, file_name="shap_report.pdf", mime="application/pdf")

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
            st.warning("⚠️ Ollama is not running. Please run `ollama run mistral`")
            st.stop()

# ---------------------------------------
# MODEL + DATA SELECTION
# ---------------------------------------

MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression",
    "KNN": "knn",
    "Decision Tree": "decision_tree",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm",
    "Linear Regression": "linear_regression"
}

def _model_and_data_selection():
    st.subheader("1. Select a model to train:")

    selected_display_name = st.selectbox("Choose a model:", list(MODEL_REGISTRY.keys()))
    module_name = MODEL_REGISTRY[selected_display_name]

    # Parámetros definidos dinámicamente por modelo
    model_params = {}
    
    if module_name == "knn":
        model_params["n_neighbors"] = st.number_input(
            "Number of Neighbors (K)", min_value=1, max_value=100, value=3, step=1
        )
    elif module_name == "decision_tree":
        model_params["max_depth"] = st.number_input(
            "Max Tree Depth", min_value=1, max_value=100, value=5, step=1
        )
    elif module_name == "svm":
        model_params["C"] = st.number_input("Penalty parameter C", min_value=0.01, value=1.0)
        model_params["kernel"] = st.selectbox("Kernel", options=["rbf", "linear", "poly", "sigmoid"])

    st.subheader("2. Upload your dataset (with target):")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file:
        data = pd.read_csv(data_file)
        st.dataframe(data.head())
        target_column = st.selectbox("Select your target column:", data.columns)

        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]

            if st.button("✨ Train model ✨"):
                try:
                    model_module = importlib.import_module(f"models.sample_models.{module_name}")
                    train_fn = model_module.train

                    model = train_fn(X, y, **model_params)

                    st.session_state.model = model
                    st.session_state.model_name = selected_display_name
                    st.session_state.data = X
                    st.session_state.target = y
                    st.session_state.analysis_started = True

                    clear_analysis_data()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to train model: {e}")

            return selected_display_name, X, y

    return None, None, None

# ---------------------------------------
# SHAP ANALYSIS + VISUALS
# ---------------------------------------

def _render_content(model_name, X, y):
    try:
        st.success(f"✅ Model loaded: {st.session_state.model_name}")
        st.success(f"✅ Dataset loaded. Shape: {X.shape}")
        st.dataframe(X.head(), use_container_width=True)

        if 'shap_values' not in st.session_state:
            with st.spinner("Calculating SHAP values..."):
                model = st.session_state.model
                explainer = ShapExplainer(model)
                shap_values = explainer.generate_shap_values(X)
                st.session_state.shap_values = shap_values

                visualizer = ShapVisualizer()
                st.session_state.plots = visualizer.create_all_plots(shap_values, X)
                st.session_state.shap_summary_img_base64 = get_img_base_64(st.session_state.plots['summary'])

        shap_values = st.session_state.shap_values
        plots = st.session_state.plots

        st.header("🔍 SHAP Analysis")
        tab1, tab2 = st.tabs(["📄 SHAP Values", "📊 Graph view"])

        with tab1:
            shap_df = pd.DataFrame(
                shap_values[0] if len(shap_values.shape) == 3 else shap_values,
                columns=X.columns
            )
            st.dataframe(shap_df.head(), use_container_width=True)

        with tab2:
            for name in ['summary', 'bar', 'beeswarm']:
                if name in plots:
                    st.markdown(f"### {name.capitalize()} Plot")
                    st.pyplot(plots[name])
                    plt.close(plots[name])

        st.header("📊 Feature Impact")
        mean_shap = np.abs(shap_values).mean(axis=0)
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=0)

        impact_df = pd.DataFrame({
            'Feature': X.columns,
            'Impact': mean_shap
        }).sort_values('Impact', ascending=False).head(5)

        tab3, tab4 = st.tabs(["📄 Table", "📊 Graph"])
        with tab3:
            st.dataframe(impact_df.style.format({'Impact': '{:.4f}'}).hide(axis='index'), use_container_width=True)
        with tab4:
            if 'importance' in plots:
                fig = plots['importance']
                fig.set_size_inches(10, 6)
                st.pyplot(fig)
                plt.close(fig)

        if 'explanation' not in st.session_state:
            with st.spinner("Generating explanation..."):
                agent = ShapAgent()
                prompt = ShapPrompts.get_analysis_prompt(model_name, shap_values, X)
                explanation = agent.generate_explanation(prompt, X.shape)
                st.session_state.explanation = explanation

        st.header("🧠 Model Insights")
        st.markdown(st.session_state.explanation)

        # PDF report
        try:
            explanation = st.session_state.explanation
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

            if st.session_state.paid:
                _render_download_button_enabled()

        except Exception as e:
            st.error("❌ Failed to generate PDF report.")
            st.exception(e)

    except Exception as e:
        st.error(f"❌ General Error: {str(e)}")
        st.error("Check dataset structure or try with fewer rows.")
