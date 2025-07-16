import re
import streamlit as st
import streamlit.components.v1 as components
from db.firebase import add_subscription
from utils.animations import set_fade_animation
from datetime import datetime

def payment_view():
    st.set_page_config(page_title="SHAP-Agent", layout="wide")
    set_fade_animation()
    if "card_number" not in st.session_state:
        st.session_state.card_number = ""
    
    #hide dev toolbar
    '''
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
""", unsafe_allow_html=True)'''
    
    ##render each individual component
    _set_custom_css()
    
    _render_inputs()
    
    _render_container()
    
    _render_confirm()
    
    
# Helper Methods 

def _set_custom_css():
    with open("shap-agent/assets/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
def _render_container():
    card_number = st.session_state.card_number
    card_holder = st.session_state.card_holder
    expires = st.session_state.expires
    
    st.markdown("""
    <div class="payment-body">
        <div class="payment-container">
            <div>
                <svg
                    id="visual"
                    viewBox="0 0 900 600"
                    xmlns="http://www.w3.org/2000/svg"
                    xmlnsXlink="http://www.w3.org/1999/xlink"
                    version="1.1"
                    class="card-svg"
                    preserveAspectRatio="none"
                >
                    <rect
                        x="0"
                        y="0"
                        width="900"
                        height="600"
                        fill="#5457CD"
                    ></rect>
                    <path
                        d="M0 400L30 386.5C60 373 120 346 180 334.8C240 323.7 300 328.3 360 345.2C420 362 480 391 540 392C600 393 660 366 720 355.2C780 344.3 840 349.7 870 352.3L900 355L900 601L870 601C840 601 780 601 720 601C660 601 600 601 540 601C480 601 420 601 360 601C300 601 240 601 180 601C120 601 60 601 30 601L0 601Z"
                        fill="#6366F1"
                        strokeLinecap="round"
                        strokeLinejoin="miter"
                   ></path>
                </svg>
                <div class="card-chip"></div>
                <div class="card-info">
                    <p class="card-number">
                        <span>&nbsp;""" + card_number[0:4] + """</span>
                        <span>""" + card_number[4:8] + """</span>
                        <span>""" + card_number[8:12] + """</span>
                        <span>""" + card_number[12:16] + """</span>
                    </p>
                    <div class="card-labels">
                        <p> Card Holder</p>
                        <p> Expires </p>
                    </div>
                    <div class="card-values">
                        <strong>""" + card_holder + """</strong>
                        <strong>""" + expires + """</strong>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
def _render_inputs():
        
    col1, col2, col3 = st.columns([4, 5, 3])
    with col2:
        st.markdown('<span id="input-after1"></span>', unsafe_allow_html=True)
        st.session_state.card_number = st.text_input("Card Number", max_chars=16)
                
        st.markdown('<span id="input-after2"></span>', unsafe_allow_html=True)
        st.session_state.card_holder = st.text_input("Card Holder:")
        
        st.markdown('<span id="input-after3"></span>', unsafe_allow_html=True)
        st.session_state.cvv = st.text_input("CVV:", type="password", max_chars=3)
        
        months = [f"{i:02d}" for i in range(1, 13)]
        years = [str(y) for y in range(datetime.now().year, datetime.now().year + 10)]
        
        st.markdown('<span id="input-after4"></span>', unsafe_allow_html=True)
        expire_month = st.selectbox("Expires On:", months, index=datetime.now().month - 1)

        st.markdown('<span id="input-after5"></span>', unsafe_allow_html=True)
        expire_year = st.selectbox(" ", years)
        
        st.session_state.expires = f"{expire_month}/{expire_year[2:4]}"
        
def _render_confirm():
    col1, col2, col3 = st.columns([4, 5, 3])
    with col2:
        st.markdown('<span id="button-after9"></span>', unsafe_allow_html=True)
        if st.button("Activate Premium", key="toggle_button"):
            if verify_input():
                add_subscription(st.session_state.time)
                st.session_state.page = "home"
                st.rerun()
            else:
                show_error()
            
def verify_input():
    card_number = st.session_state.card_number
    card_holder = st.session_state.card_holder
    cvv = st.session_state.cvv
    valid = True
    error = ""
    
    if not card_number or card_number == "":
        error = "Card number cannot be empty."
        valid = False
    
    if not card_number.isdigit():
        error = "Card number must contain only numbers."
        valid = False

    elif len(card_number) != 16:
        error = "Card number must be exactly 16 digits."
        valid = False
        
    elif not card_holder or card_holder == "":
        error = "Card holder cannot be empty."
        valid = False
        
    elif not cvv or cvv == "":
        error = "CVV cannot be empty."
        valid = False

    elif not cvv.isdigit():
        error = "CVV must contain only numbers."
        valid = False

    elif len(cvv) != 3:
        error = "CVV must be exactly 3 digits."
        valid = False

        
    if valid == False:
        st.session_state.error = error
        
    return valid

def show_error():
    st.markdown("""
    <div class="error-div">
       <p>""" + st.session_state.error +"""</p>
    </div>
    """, unsafe_allow_html=True)
