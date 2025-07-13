import base64
import io
import streamlit as st


def get_img_base_64(figure):
    if figure is not None:
        buf = io.BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_bytes = buf.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')          
        return img_base64
    return None

# delete all analysis data from session state
def clear_analysis_data():
    for key in ['explanation', 'plots', 'data', 'model', 'model_name', 'shap_values', 'pdf_bytes']:
        st.session_state.pop(key, None)
                    