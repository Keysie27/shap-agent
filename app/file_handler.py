# Utility functions to load uploaded models and datasets

import pandas as pd
import joblib
import pickle
import tempfile


def load_model(uploaded_file):
    """
    Load a machine learning model from a .pkl or .joblib file.
    """
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    if suffix == "pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif suffix == "joblib":
        return joblib.load(path)
    else:
        raise ValueError("Unsupported file format. Please upload .pkl or .joblib")


def load_dataset(uploaded_file):
    """
    Load a dataset from a CSV file as a pandas DataFrame.
    """
    return pd.read_csv(uploaded_file)