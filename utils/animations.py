import streamlit as st

def set_fade_animation():
    st.markdown("""
    <style>
    .block-container {
        animation: fadeIn 0.6s ease-in-out;
    }

    @keyframes fadeIn {
        0%   { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
