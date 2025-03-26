# Streamlit UI logic: file upload, SHAP explanation, agent output

import streamlit as st
from app.file_handler import load_model, load_dataset
from utils.shap_explainer import generate_shap_summary
from agent.shap_agent import explain_with_agent, check_ollama_alive

st.set_page_config(page_title="SHAP-Agent", layout="wide")
st.title("🤖 SHAP-Agent: Explain Your Model with SHAP + LLM")

if not check_ollama_alive():
    st.warning("⚠️ Ollama is not running. Please run `ollama run mistral` in your terminal.")

st.markdown("""
Upload your machine learning model and dataset. We'll generate a global SHAP summary 
and use a local LLM agent (like Mistral) to explain the model's behavior in natural language.
""")

model_file = st.file_uploader("Upload your model (.pkl or .joblib)", type=["pkl", "joblib"])
data_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if model_file and data_file:
    with st.spinner("Loading model and dataset..."):
        model = load_model(model_file)
        data = load_dataset(data_file)

        if data.empty:
            st.error("❌ The uploaded dataset is empty.")
            st.stop()
        if not hasattr(model, "predict"):
            st.error("❌ The uploaded model is not valid. It must implement .predict().")
            st.stop()

    with st.spinner("Generating SHAP summary..."):
        try:
            shap_summary = generate_shap_summary(model, data)
        except Exception as e:
            st.error(f"❌ SHAP failed to explain the model:\n{e}")
            st.stop()

    with st.spinner("Generating natural language explanation with agent..."):
        explanation = explain_with_agent(shap_summary, data.shape)

    st.success("✅ Explanation Ready")
    st.subheader("🧠 Natural Language Explanation")
    st.write(explanation)

    st.download_button("📄 Download Explanation", explanation, file_name="shap_explanation.txt")
