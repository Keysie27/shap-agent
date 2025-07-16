from views.home import home_view
from views.plans import plans_view
from views.payment import payment_view
from views.mode_selector import mode_selector_view
from db.firebase import verify_key, get_subscription_by_key
import streamlit as st

def main():
    import warnings
    warnings.filterwarnings("ignore")

    data = get_subscription_by_key()
    valid = verify_key(data)
    st.session_state.paid = valid

    if "page" not in st.session_state:
        st.session_state.page = "mode_selector"

    if st.session_state.page == "plans":
        plans_view()
    elif st.session_state.page == "mode_selector":
        mode_selector_view()
    elif st.session_state.page == "payment":
        payment_view()
    elif st.session_state.page == "home":
        home_view()

if __name__ == "__main__":
    main()
