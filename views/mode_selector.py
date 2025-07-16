import streamlit as st

def mode_selector_view():
    st.set_page_config(page_title="SHAP-Agent | Choose Mode", layout="centered")

    st.title("🔍 Welcome to SHAP-Agent")
    st.markdown("Select how deeply you'd like to explore your model explanation.")

    st.markdown('<div style="text-align:right;">'
                '<button style="background-color:#6f42c1;color:white;border:none;padding:0.5em 1em;border-radius:8px;cursor:pointer;" '
                'onclick="window.location.href=\'#\'">💎 Get Premium</button>'
                '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Smart Insights (Standard)", use_container_width=True):
            st.session_state.explanation_mode = "standard"
            st.session_state.page = "home"
            st.rerun()

    with col2:
        # Check if the user has a paid subscription
        if st.session_state.get("paid", False):
            if st.button("🚀 Pro Insights (Advanced)", use_container_width=True):
                st.session_state.explanation_mode = "advanced"
                st.session_state.page = "home"
                st.rerun()
        else:
            st.button("🚀 Pro Insights (Advanced)", disabled=True, use_container_width=True)
            st.info("🔒 Pro Insights is only available for Premium users.")
