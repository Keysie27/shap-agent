import re
import os
import importlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from services.subscription_handler import update_field, load_data
from agent.shap_agent import ShapAgent
from services.helpers import clear_analysis_data, get_img_base_64
from services.pdf_generator import create_shap_report_pdf
from shap_tools.explainer import ShapExplainer
from shap_tools.visualizations import ShapVisualizer
from agent.prompts import ShapPrompts
from utils.animations import set_fade_animation

# Model registry
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm",
    "Linear Regression": "linear_regression"
}

def home_view():
    st.set_page_config(page_title="Whitebox XAI Agent", layout="wide")

    set_fade_animation()

    _render_toggle_button()

    if st.session_state.get('paid', False):
        st.markdown("""
            <div style="background-color: #198754; color: white; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                ✅ You're using the <strong>Premium</strong> version of Whitebox XAI Agent!
            </div>
        """, unsafe_allow_html=True)

    _hide_toolbar()
        
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

    st.subheader("2. Upload your training dataset:")
    data_file = st.file_uploader("Upload CSV (must include the target column):", type=["csv"])

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

        #sample data for pdf
        sample_data_list = [data.columns.to_list()] + data.head(5).values.tolist()

        st.markdown(" 👀 Take a look at your data:")
        st.dataframe(data.head(3))
        st.write(f"Total rows: {data.shape[0]}, Total columns: {data.shape[1]}")
        target_column = st.selectbox("❗Select the target column:", data.columns)

        st.subheader("3. Enter data to test your model:")
        input_columns = [col for col in data.columns if col != target_column]

        # Matrix for test input
        default_row = {
            col: data[col].median() if np.issubdtype(data[col].dtype, np.number) else data[col].mode().iloc[0]
            for col in input_columns
        }
        test_input_df = pd.DataFrame([default_row])

        st.markdown("✍️ Edit your test input values below:")
        edited_test_input = st.data_editor(test_input_df, use_container_width=True, key="input_matrix")

        true_label_input = st.text_input("🎯 Enter expected output:")

        update_field("date", datetime.now().date().isoformat())

        if st.session_state.paid or load_data()['count'] < 3:
            if st.button("✨ Explain my model ✨", use_container_width=True):
                st.session_state.loading = True
                
                clear_analysis_data()
                st.markdown("## Training and SHAP Analysis")

                try:
                    with st.spinner("🔍 Training model and generating SHAP values..."):
                        X_raw = data.drop(columns=[target_column])
                        y_raw = data[target_column]
                        X_numeric = pd.get_dummies(X_raw, drop_first=False).astype(float)
                        if X_numeric.shape[1] == 0:
                            st.error("❌ No usable numeric features for SHAP.")
                            return

                        if module_name == "linear_regression":
                            y = y_raw  # keep numeric values
                        else:
                            label_encoder = LabelEncoder()
                            y = label_encoder.fit_transform(y_raw)
                            class_names = label_encoder.classes_ # Get class names for predictions for future use

                        model_module = importlib.import_module(f"models.sample_models.{module_name}")
                        model = model_module.train(X_numeric, y, **model_params)

                        test_df = edited_test_input.copy()
                        test_df = test_df.reindex(columns=X_numeric.columns, fill_value=0).astype(float)

                        waterfall_input = test_df

                        prediction = model.predict(test_df)[0]
                        
                        if module_name == "linear_regression":
                            predicted_label = prediction
                        else:
                            predicted_label = label_encoder.inverse_transform([prediction])[0]

                        st.write(f"🔮 **Model Prediction for Input:** `{predicted_label}`")

                        if true_label_input:
                            try:
                                if module_name == "linear_regression":
                                    true_val = float(true_label_input)
                                    error = np.abs(true_val - prediction)
                                    st.info(f"📏 Absolute Error: **{error:.4f}**")
                                else:
                                    true_val = label_encoder.transform([true_label_input])[0]
                                    is_correct = int(true_val == prediction)
                                    acc = is_correct * 100
                                    st.info(f"📈 Accuracy: **{acc:.0f}%**")
                            except Exception as e:
                                st.warning(f"⚠️ Couldn't compute accuracy. Check if the input is valid: {e}")

                        explainer = ShapExplainer(model, background_data=X_numeric)
                        explainer.create_explainer()
                        
                        shap_values = explainer.generate_shap_values(X_numeric)
                        test_shap_values = explainer.generate_shap_values(test_df)

                        visualizer = ShapVisualizer()
                        plots = visualizer.create_all_plots(shap_values, X_numeric, test_shap_values, waterfall_input, feature_names=None, waterfall_base_value=0)

                        plt.style.use('dark_background')
                        if 'importance' in plots:
                            fig = plots['importance']
                            fig.set_size_inches(8, 4)
                            fig.patch.set_facecolor('#0e1117')
                            fig.patch.set_alpha(0.8)
                            ax = fig.axes[0]
                            ax.set_facecolor("#0e1117")
                            ax.title.set_color('white')
                            ax.xaxis.label.set_color('white')
                            ax.yaxis.label.set_color('white')
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                            ax.set_xlabel("Impact in the model", fontsize=16, color='white', labelpad=10)
                            ax.set_ylabel("Feature", fontsize=16, color='white', labelpad=10)
                            for bar in ax.patches:
                                bar.set_color("#8EA0F0")
    
                        summary_base64 = get_img_base_64(plots['summary']) if 'summary' in plots else None
                        bar_base64 = get_img_base_64(plots['importance']) if 'importance' in plots else None
                        waterfall_base64 = get_img_base_64(plots['waterfall'], "waterfall") if 'waterfall' in plots else None


                        st.session_state.update({
                            'model_name': selected_display_name,
                            'shap_values': shap_values,
                            'plots': plots,
                            'shap_summary_img_base64': summary_base64
                        })

                        st.success("✅ Model trained and SHAP analysis complete!")

                        tab1, tab2 = st.tabs(["Feature Impact", "Shapley Values Overview"])
                        with tab1:
                            st.pyplot(plots['importance']) if 'importance' in plots else None
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
                        is_advanced = st.session_state.get("explanation_mode") == "advanced"

                        if is_advanced:
                            prompt = ShapPrompts.get_advanced_analysis_prompt(selected_display_name, shap_values, X_numeric)
                        else:
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
                        practical_recommendations=recommendations,
                        sample_data=sample_data_list,
                        shap_values=shap_df,
                        waterfall_img_base64=waterfall_base64
                    )
                    
                    if st.session_state.loading == True:
                        #when finished analysis, add to counter
                        update_field("count")
                        st.session_state.loading = False

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.session_state.get('paid', False):
                        st.success("✅ XAI Advanced report is ready!")
                        st.download_button("📥 Download XAI report", pdf_bytes, "shap_report.pdf", mime="application/pdf")
                    else:
                        st.markdown("""
                            <button class="disabled-button" disabled>📥 Advanced XAI Report</button>
                        """, unsafe_allow_html=True)
                        st.info("🔒  Advanced XAI report is available for premium users only.")

                except Exception as e:
                    st.error(f"❌ Report Generation Error: {e}")

        else:
            st.markdown("""
                <button class="disabled-button" style="width: 100%" disabled> Explain my model </button>
            """, unsafe_allow_html=True)
            st.info("🔒  Daily limit reached on the free plan. Upgrade for unlimited use.")

def _check_ollama():
    with st.spinner("Checking Ollama service..."):
        if not ShapAgent.check_ollama_alive():
            st.error("⚠️ Ollama is not running. Please run `ollama run mistral`")
            st.stop()

# UI elements

def _set_custom_css():
    with open("shap-agent/assets/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _render_header():
    st.title("💡 WhiteBox XAI Agent: AI Model Explanation")
    st.markdown("Understand how your ML model makes decisions!")

    # Show selected mode
    mode = st.session_state.get("explanation_mode", "standard")

    if mode == "advanced":
        st.markdown("""
            <div style="border: 2px solid #6f42c1; color: white; padding: 0.7rem 1rem; border-radius: 10px; margin-top: 0.5rem; margin-bottom: 1.5rem; font-size: 16px;">
                🚀 <strong>Pro Insights (Advanced)</strong> mode is active.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="border: 2px solid #6f42c1; color: white; padding: 0.7rem 1rem; border-radius: 10px; margin-top: 0.5rem; margin-bottom: 1.5rem; font-size: 16px;">
                🧠 <strong>Smart Insights (Standard)</strong> mode is active.
            </div>
        """, unsafe_allow_html=True)

def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        ### Instructions:
        1. Choose a model
        2. Upload CSV with target column
        3. Enter test data
        4. Click on ✨Explain my model✨
        ---
        ### Output includes:
        - Feature impact analysis
        - Visualizations
        - AI-powered explanation
        - Advanced XAI report
        """)

def _render_toggle_button():
    col1, col2 = st.columns([2, 16])

    with col1:
        st.markdown('<span id="button-after11"></span>', unsafe_allow_html=True)
        if st.button("⬅ Welcome", key="back_btn", help="Go to welcome page"):
            st.session_state.page = "mode_selector"
            st.rerun()

    with col2:
        st.markdown('<span id="button-after11"></span>', unsafe_allow_html=True)
        if st.button("💎 Upgrade to Premium", key="premium_btn", help="Go to plans"):
            st.session_state.page = "plans"
            st.rerun()

def _hide_toolbar():
    st.markdown("""
    <style>
    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)