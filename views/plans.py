import re
import streamlit as st
import streamlit.components.v1 as components
from db.firebase import add_subscription
from utils.animations import set_fade_animation

pdf_bytes = None

def plans_view():
    st.set_page_config(page_title="SHAP-Agent", layout="wide")
    set_fade_animation()

    if "price" not in st.session_state:
      st.session_state.price = "12.99"
      st.session_state.time = "month"

    #hide dev toolbar
    #'''
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
""", unsafe_allow_html=True)#'''
    
    ##render each individual component
    _set_custom_css()

    _render_header()
    
    _render_time_toggle()
    
    _render_prices()
    
    _render_free_button()
    
    _render_paid_button()

    
# Helper Methods 

def _set_custom_css():
    with open("shap-agent/assets/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
def _render_header():
    st.markdown("""
    <div class="price-header">
        <h1>Plans and Pricing</h1>
        <p>Choose a plan tailored to your needs</p>
    </div>
    """, unsafe_allow_html=True)
    
def _render_time_toggle():
    st.markdown('<span id="button-after7"></span>', unsafe_allow_html=True)
    toggle = st.toggle(" ")
    
    if toggle: 
        st.session_state.price = "79.99"
        st.session_state.time = "year"
    else:
        st.session_state.price = "12.99"
        st.session_state.time = "month"
    
    st.markdown('<span id="div-after1"></span>', unsafe_allow_html=True)
    st.markdown('''
    <div class="price-toggle-labels">
        <div class="toggle-label-1">Monthly</div>
        <div class="toggle-label-2">Yearly</div>
    </div>
    ''', unsafe_allow_html=True)

def _render_prices():
# HTML price container
    price = st.session_state.price
    selectedTime = st.session_state.time
    
    st.markdown('''
    <div class="price-root">
        <div class="price-container1">
            <p class="price-title">Standard</p>
            <p class="price-cost">Free</p>
            <div class="price-line1"></div>
            <div class="price-list-div">
                <?xml version="1.0" encoding="utf-8"?><svg class="price-star1" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88" style="enable-background:new 0 0 122.88 122.88" xml:space="preserve"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path class="st0" d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"/></g></svg>
                <p class="price-text1">Three <span> daily explanations</span></p>
            </div>
            <div class="price-list-div">
                <?xml version="1.0" encoding="utf-8"?><svg class="price-star1" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88" style="enable-background:new 0 0 122.88 122.88" xml:space="preserve"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path class="st0" d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"/></g></svg>
                <p class="price-text1"><span>Predefined </span>models</p>
            </div>
        </div>
        <div class="price-container2">
            <p class="price-title">Premium</p>
            <p class="price-cost">$''' + str(price) + '''<span> / ''' + selectedTime + '''</span></p>
            <div class="price-line2"></div>
            <div class="price-list-div">
                <?xml version="1.0" encoding="utf-8"?><svg class="price-star2" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88" style="enable-background:new 0 0 122.88 122.88" xml:space="preserve"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path class="st0" d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"/></g></svg>
                <p class="price-text2">Unlimited <span>explanations</span></p>
            </div>
            <div class="price-list-div">
                <?xml version="1.0" encoding="utf-8"?><svg class="price-star2" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88" style="enable-background:new 0 0 122.88 122.88" xml:space="preserve"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path class="st0" d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"/></g></svg>
                <p class="price-text2"><span>Upload </span>models</p>
            </div>
            <div class="price-list-div">
                <?xml version="1.0" encoding="utf-8"?><svg class="price-star2" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88" style="enable-background:new 0 0 122.88 122.88" xml:space="preserve"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path class="st0" d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"/></g></svg>
                <p class="price-text2">Download <span>reports</span></p>
            </div>
            <div class="price-list-div">
                <?xml version="1.0" encoding="utf-8"?><svg class="price-star2" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88" style="enable-background:new 0 0 122.88 122.88" xml:space="preserve"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path class="st0" d="M62.43,122.88h-1.98c0-16.15-6.04-30.27-18.11-42.34C30.27,68.47,16.16,62.43,0,62.43v-1.98 c16.16,0,30.27-6.04,42.34-18.14C54.41,30.21,60.45,16.1,60.45,0h1.98c0,16.15,6.04,30.27,18.11,42.34 c12.07,12.07,26.18,18.11,42.34,18.11v1.98c-16.15,0-30.27,6.04-42.34,18.11C68.47,92.61,62.43,106.72,62.43,122.88L62.43,122.88z"/></g></svg>
                <p class="price-text2"><span>Advanced </span>explanations</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
def _render_free_button():
    st.markdown('<span id="button-after5"></span>', unsafe_allow_html=True)
    if st.button("Get started", key="free_button"):
        st.session_state.page = "home"
        st.rerun()

def _render_paid_button():
    st.markdown('<span id="button-after6"></span>', unsafe_allow_html=True)
    if st.button("Upgrade to Premium", key="paid_button"):
        st.session_state.page = "payment"
        st.rerun()
