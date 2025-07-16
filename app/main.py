from views.home import home_view
from views.plans import plans_view
from views.payment import payment_view
from db.firebase import verify_key, get_subscription_by_key
import streamlit as st

def main():
    import warnings
    warnings.filterwarnings("ignore")
            
    #verify if user is logged in
    data = get_subscription_by_key()
    valid = verify_key(data)
    st.session_state.paid = valid
        
    #check which view to load
    
    if "page" not in st.session_state or st.session_state.page == "home":
        home_view()
    
    elif st.session_state.page == "plans":
        plans_view()
        
    elif st.session_state.page == "payment":
        payment_view()

if __name__ == "__main__":
    main()