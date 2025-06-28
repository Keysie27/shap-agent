import re
import streamlit as st
from views.home import home_view
from services.pdf_generator import create_shap_report_pdf

def main():
    import warnings
    warnings.filterwarnings("ignore")

    home_view()

if __name__ == "__main__":
    main()