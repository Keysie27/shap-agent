import re
import os
import importlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from agent.shap_agent import ShapAgent
from services.helpers import clear_analysis_data, get_img_base_64
from services.pdf_generator import create_shap_report_pdf
from shap_tools.explainer import ShapExplainer
from shap_tools.visualizations import ShapVisualizer
from agent.prompts import ShapPrompts

# Model registry
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
    _render_toggle_button()
    _set_custom_css()
    _render_header()
    _render_sidebar()
    _check_ollama()

    st.subheader("1. Select a model:")
    selected_display_name = st.selectbox("Choose a model:", list(MODEL_REGISTRY.keys()), key="model_select")
    module_name = MODEL_REGISTRY[selected_display_name]

    model_params = {}
    if module_name == "knn":
        model_params["n_neighbors"] = st.number_input("Number of Neighbors (K)", min_value=1, max_value=100, value=3)
    elif module_name == "decision_tree":
        model_params["max_depth"] = st.number_input("Max Depth", min_value=1, max_value=100, value=5)

    st.subheader("2. Upload your dataset:")
    data_file = st.file_uploader("Upload CSV (must include a target column):", type=["csv"])

    purple_button_style = """
        <style>
        div.stButton > button:first-child {
            background-color: #6f42c1 !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #5a32a3 !important;
            transform: scale(1.02);
        }
        </style>
    """
    st.markdown(purple_button_style, unsafe_allow_html=True)

    if data_file:
        data = pd.read_csv(data_file)
        st.markdown(" 👀 Take a look at your data:")
        st.dataframe(data.head(3))
        st.write(f"Total rows: {data.shape[0]}, Total columns: {data.shape[1]}")
        target_column = st.selectbox("Select the target column:", data.columns)

        if st.button("✨ Explain my model ✨", use_container_width=True):
            clear_analysis_data()
            st.markdown("## Training and SHAP Analysis")

            try:
                with st.spinner("🔍 Training model and generating SHAP values..."):
                    X_raw = data.drop(columns=[target_column])
                    y = data[target_column]
                    # Convert categorical variables into dummy/indicator variables
                    X_numeric = pd.get_dummies(X_raw, drop_first=False)
                    X_numeric = X_numeric.astype(float)
                    if X_numeric.shape[1] == 0:
                        st.error("❌ No usable numeric features for SHAP.")
                        return

                    model_module = importlib.import_module(f"models.sample_models.{module_name}")
                    model = model_module.train(X_numeric, y, **model_params)

                    explainer = ShapExplainer(model)
                    shap_values = explainer.generate_shap_values(X_numeric)

                    visualizer = ShapVisualizer()
                    plots = visualizer.create_all_plots(shap_values, X_numeric)
                    plt.style.use('dark_background')
                    if 'importance' in plots:
                        fig = plots['importance']
                        fig.set_size_inches(8, 4)
                        fig.patch.set_facecolor('#0e1117')  # Dark background
                        fig.patch.set_alpha(0.8)  # Slight transparency
                        ax = fig.axes[0]
                        ax.set_facecolor("#0e1117") # Background for the axes

                        # Set axis labels and title colors
                        ax.title.set_color('white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.tick_params(axis='x', colors='white')
                        ax.tick_params(axis='y', colors='white')
                        ax.set_xlabel("Impact in the model", fontsize=16, color='white', labelpad=10)
                        ax.set_ylabel("Features", fontsize=16, color='white', labelpad=10)

                        # Style bars
                        for bar in ax.patches:
                            bar.set_color("#8EA0F0")  # Color for bars
                            # bar.set_alpha(0.4) # Slight transparency for bars

                    summary_base64 = get_img_base_64(plots['summary']) if 'summary' in plots else None
                    bar_base64 = get_img_base_64(plots['importance']) if 'importance' in plots else None

                    st.session_state.update({
                        'model_name': selected_display_name,
                        'shap_values': shap_values,
                        'plots': plots,
                        'shap_summary_img_base64': summary_base64
                    })

                    st.success("✅ Model trained and SHAP analysis complete!")

                    tab1, tab2 = st.tabs(["Feature Impact", "Raw SHAP Values"])
                    with tab1:
                        st.pyplot(plots['importance'])
                    with tab2:
                        shap_df = pd.DataFrame(
                            shap_values[0] if len(shap_values.shape) == 3 else shap_values,
                            columns=X_numeric.columns
                        )
                        st.dataframe(shap_df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ SHAP Analysis Error: {e}")
                return

            st.markdown("## AI Explanation")
            try:
                with st.spinner("✨ Generating AI explanation..."):
                    agent = ShapAgent()
                    prompt = ShapPrompts.get_analysis_prompt(selected_display_name, shap_values, X_numeric)
                    explanation = agent.generate_explanation(prompt, X_numeric.shape)
                    st.session_state['explanation'] = explanation
                    st.markdown(explanation)
            except Exception as e:
                st.error(f"❌ Explanation Error: {e}")
                return

            try:
                sections = re.split(r"\*\*\d+\.\s?", explanation.strip())
                cleaned = [s.replace('*', '').strip() for s in sections]

                summary = cleaned[1] if len(cleaned) >= 2 else "No summary available."
                top_features = [s.strip() for s in re.split(r"- (?=Feature Name|\*\*|[\w])", cleaned[2])] if len(cleaned) >= 3 else []
                observations = [s.strip() for s in cleaned[3].split('- ')] if len(cleaned) >= 4 else []
                recommendations = [s.strip() for s in cleaned[4].split('- ')] if len(cleaned) >= 5 else []

                pdf_bytes = create_shap_report_pdf(
                    shap_summary_img_base64=summary_base64,
                    bar_chart_img_base64=bar_base64,
                    top_influencers_sentence=summary,
                    feature_analysis_points=top_features,
                    key_observations_points=observations,
                    practical_recommendations=recommendations
                )

                st.markdown("<br>", unsafe_allow_html=True)
                if st.session_state.get('paid', False):
                    st.success("✅ XAI Advanced report is ready!")
                    st.download_button("📥 Download XAI report", pdf_bytes, "shap_report.pdf", mime="application/pdf")
                else:
                    st.markdown("""
                        <style>
                        .disabled-button {
                            background-color: #cccccc !important;
                            color: #666666 !important;
                            cursor: not-allowed !important;
                            border-radius: 8px;
                            padding: 0.5em 1em;
                            font-weight: bold;
                            border: none;
                        }
                        </style>
                        <button class="disabled-button" disabled>📥 Advanced XAI Report</button>
                    """, unsafe_allow_html=True)
                    st.info("🔒  Advanced XAI report is available for premium users only.")

            except Exception as e:
                st.error(f"❌ Report Generation Error: {e}")

# UI elements

def _set_custom_css():
    with open("shap-agent/assets/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _render_toggle_button():
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    if st.button("💎 Get Premium", key="toggle_button"):
        st.session_state.page = "payment"
        st.rerun()

def _render_header():
    st.title("💡 SHAP-Agent: AI Model Explanation")
    st.markdown("Understand how your ML model makes decisions!")

def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        ### Instructions:
        1. Choose a model
        2. Upload CSV with target column
        3. Click '✨ Explain my model ✨'
        ---
        Output includes:
        - Feature impact analysis
        - SHAP visualizations
        - AI-powered explanation
        - Advanced XAI report
        """)

def _check_ollama():
    with st.spinner("Checking Ollama service..."):
        if not ShapAgent.check_ollama_alive():
            st.error("⚠️ Ollama is not running. Please run `ollama run mistral`")
            st.stop()
