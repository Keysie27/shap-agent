import re
import streamlit as st
from views.paid_views.main_view import main_view
from services.pdf_generator import create_shap_report_pdf

def main():
    import warnings
    warnings.filterwarnings("ignore")

    main_view()

if __name__ == "__main__":
    main()