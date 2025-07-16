import streamlit as st
from utils.animations import set_fade_animation

def mode_selector_view():
    st.set_page_config(page_title="SHAP-Agent | Choose Mode", layout="centered")
    set_fade_animation()
    _set_custom_css()

    st.markdown("<h1 style='text-align: center;'>💡 Welcome to <span style='color:#6f42c1;'>SHAP-Agent</span>!</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 18px; padding-bottom: 1em;'>
        <strong>Choose how deeply you'd like to explore your model's behavior with AI explanations!</strong><br><br>
        Select one of the available modes below to begin your interpretability journey:
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.button("🧠 Smart Insights (Standard)", key="standard_mode", use_container_width=True,
                on_click=_select_mode, kwargs={"mode": "standard"})

        st.markdown("""
        <div class="checkpoints">
            ✅ Up to 3 explanations per day<br>
            ✅ Business-friendly SHAP summaries<br>
            ✅ Fast and lightweight insights
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.session_state.get("paid", False):
            st.button("🚀 Pro Insights (Advanced)", key="advanced_mode", use_container_width=True,
                    on_click=_select_mode, kwargs={"mode": "advanced"})
        else:
            st.button("🚀 Pro Insights (Advanced)", disabled=True, use_container_width=True)

        st.markdown("""
        <div class="checkpoints">
            ✅ Unlimited explanations<br>
            ✅ Technical & business deep-dive<br>
            ✅ PDF report with advanced visualizations
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.get("paid", False):
            st.markdown("""
            <div style='background-color:#1c2733; padding: 1em; border-radius: 10px; margin-top: 0.5em;'>
                <span style='color: #90caf9;'>🔒 Pro Insights is only available for <strong>Premium users</strong>.</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _render_toggle_button()


def _render_toggle_button():
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        if st.button("💎 Upgrade to Premium", key="toggle_button", use_container_width=True):
            st.session_state.page = "plans"
            st.rerun()

def _select_mode(mode: str):
    st.session_state.explanation_mode = mode
    st.session_state.page = "home"


def _set_custom_css():
    st.markdown("""
    <style>
    /* Transparent buttons (Smart & Pro) */
    div.stButton > button:not(#toggle_button) {
        background-color: transparent;
        color: white;
        font-weight: bold;
        border: 2px solid #6f42c1;
        border-radius: 12px;
        padding: 0.75em 1.5em;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    div.stButton > button:not(#toggle_button):hover {
        background-color: rgba(111, 66, 193, 0.1);
        border-color: #a277ff;
        box-shadow: 0 0 10px #6f42c1;
        transform: scale(1.03);
    }

    /* Purple Premium button */
    div.stButton > button#toggle_button {
        background-color: #6f42c1 !important;
        color: white !important;
        font-weight: bold;
        border: none;
        font-size: 18px;
        padding: 0.8em 1.5em;
        border-radius: 12px;
        transition: 0.3s ease;
    }

    div.stButton > button#toggle_button:hover {
        background-color: #5a32a3 !important;
        transform: scale(1.02);
    }

    .checkpoints {
        padding-top: 0.5em;
        font-size: 15px;
        line-height: 1.7;
    }

    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
