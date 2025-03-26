# Test loading models and CSVs from sample folders

import os
from app.file_handler import load_model, load_dataset

def test_load_model():
    path = "sample_models/random_forest.pkl"
    assert os.path.exists(path), "Sample model file does not exist"
    with open(path, "rb") as f:
        model = load_model(f)
        assert hasattr(model, "predict"), "Loaded model must have a predict method"

def test_load_dataset():
    path = "data/sample_data/random_forest.csv"
    assert os.path.exists(path), "Sample CSV does not exist"
    with open(path, "rb") as f:
        df = load_dataset(f)
        assert df.shape[0] > 0, "Dataset should not be empty"
