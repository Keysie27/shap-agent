import pandas as pd
import joblib
import pickle
from io import BytesIO

def load_dataset(uploaded_file):
    """Robust dataset loader that handles both paths and Streamlit UploadedFile"""
    try:
        if hasattr(uploaded_file, 'read'):  # Streamlit UploadedFile
            return pd.read_csv(BytesIO(uploaded_file.getvalue()))
        elif isinstance(uploaded_file, str):  # File path
            return pd.read_csv(uploaded_file)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")