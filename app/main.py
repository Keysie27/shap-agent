# Streamlit UI logic: file upload, SHAP explanation, agent output
import streamlit as st
import shap
import os
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from app.file_handler import load_dataset
from utils.shap_explainer import generate_shap_summary
from agent.shap_agent import explain_with_agent, check_ollama_alive

# Configuración de la página
st.set_page_config(page_title="SHAP-Agent", layout="wide")
st.title("🤖 SHAP-Agent: Explain Your Model with SHAP + LLM")

# Verificar Ollama
if not check_ollama_alive():
    st.warning("⚠️ Ollama no está corriendo. Por favor ejecuta `ollama run mistral` en tu terminal.")

# Opciones de modelo
MODEL_OPTIONS = {
    "Regresión Logística": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Elementos de UI
selected_model_name = st.selectbox(
    "Selecciona un modelo para analizar:",
    list(MODEL_OPTIONS.keys())
)

data_file = st.file_uploader("Sube tu dataset (.csv)", type=["csv"])

if selected_model_name and data_file:
    with st.spinner("Cargando modelo y dataset..."):
        try:
            # Cargar modelo
            model_path = os.path.join("models", "sample_models", MODEL_OPTIONS[selected_model_name])
            model = joblib.load(model_path)
            
            # Validar modelo
            if not hasattr(model, "predict"):
                st.error("❌ El modelo cargado no es válido. Debe tener método predict().")
                st.stop()
            
            # Cargar y validar dataset
            try:
                data = load_dataset(data_file)
                
                # Mostrar vista previa
                st.write("📄 Vista previa del dataset:", data.head())
                
                # Verificar tipos de datos
                if not all(pd.api.types.is_numeric_dtype(dt) for dt in data.dtypes):
                    st.warning("⚠️ Se detectaron columnas no numéricas. Convirtiendo...")
                    data = data.apply(pd.to_numeric, errors='coerce')
                    if data.isnull().any().any():
                        st.error("❌ No se pudieron convertir todas las columnas a numéricas")
                        st.stop()
                
                # Eliminar target si existe
                if 'target' in data.columns:
                    data = data.drop(columns=['target'])
                
                st.success(f"✅ Modelo cargado: {selected_model_name}")
                st.success(f"✅ Dataset cargado. Forma: {data.shape}")
                
            except Exception as e:
                st.error(f"❌ Error al cargar el dataset: {str(e)}")
                st.stop()

            # Generar explicación SHAP (con manejo mejorado)
            with st.spinner("Generando resumen SHAP..."):
                try:
                    # Llamada mejorada a SHAP
                    explainer = shap.Explainer(model, data)
                    shap_values = explainer(data)
                    
                    # Manejo de diferentes tipos de salida SHAP
                    if hasattr(shap_values, 'values'):
                        shap_values = np.abs(shap_values.values)
                    else:
                        shap_values = np.abs(shap_values)
                    
                    # Calcular importancia media
                    mean_shap = pd.Series(np.mean(shap_values, axis=0))
                    mean_shap.index = data.columns
                    mean_shap = mean_shap.sort_values(ascending=False)
                    
                    # Formatear resumen
                    shap_summary = "Top 10 características importantes:\n\n"
                    for feature, value in mean_shap.head(10).items():
                        shap_summary += f"- {feature}: {value:.4f}\n"
                        
                    st.success("✅ SHAP calculado correctamente")
                    
                except Exception as e:
                    st.error(f"❌ Error en SHAP: {str(e)}")
                    st.stop()

            # Generar explicación en lenguaje natural
            with st.spinner("Generando explicación con IA..."):
                try:
                    explanation = explain_with_agent(shap_summary, data.shape)
                except Exception as e:
                    st.error(f"❌ Error al generar explicación: {str(e)}")
                    st.stop()

            # Mostrar resultados
            st.success("✅ Explicación lista")
            st.subheader("🧠 Explicación del Modelo")
            st.write(explanation)

            # Botón de descarga
            st.download_button(
                "📥 Descargar Explicación",
                explanation,
                file_name=f"explicacion_{selected_model_name.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error crítico: {str(e)}")
            st.stop()

# Sección de información adicional
st.sidebar.markdown("""
### ℹ️ Instrucciones:
1. Selecciona un modelo pre-entrenado
2. Sube tu dataset en formato CSV
3. Espera a que se genere la explicación

### 📌 Notas:
- Los modelos deben estar en la carpeta `models/sample_models/`
- El dataset debe contener solo características (sin columna target)
- Las columnas no numéricas se convertirán automáticamente
""")