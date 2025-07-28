import base64
import io
import streamlit as st
import pickle

def clone_figure(fig):
    buf = io.BytesIO()
    pickle.dump(fig, buf)
    buf.seek(0)
    return pickle.load(buf)

def get_img_base_64(figure, type=None):
    
    if figure is not None:
        buf = io.BytesIO()

        fig_copy = clone_figure(figure)

        fig_copy.patch.set_facecolor('#ffffff')
        fig_copy.patch.set_alpha(0.8)
        ax = fig_copy.axes[0]
        ax.set_facecolor("#ffffff")
        ax.title.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.set_xlabel("Impact in the model", fontsize=16, color='black', labelpad=10)
        ax.set_ylabel("Feature", fontsize=16, color='black', labelpad=10)

        if type == "waterfall":
            ""
        else:
            for bar in ax.patches:
                bar.set_color("#5C76E7")

        fig_copy.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_bytes = buf.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')          
        return img_base64
    return None

# delete all analysis data from session state
def clear_analysis_data():
    for key in ['explanation', 'plots', 'data', 'model', 'model_name', 'shap_values', 'pdf_bytes']:
        st.session_state.pop(key, None)
                    
