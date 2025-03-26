# Ensure SHAP explanation works with a known model and dataset

from utils.shap_explainer import generate_shap_summary
from app.file_handler import load_model, load_dataset

def test_shap_summary_from_random_forest():
    with open("sample_models/random_forest.pkl", "rb") as model_file, \
         open("data/sample_data/random_forest.csv", "rb") as data_file:
        model = load_model(model_file)
        data = load_dataset(data_file)

        summary = generate_shap_summary(model, data)
        assert "Top 10 features" in summary, "SHAP summary format invalid"
        assert len(summary.splitlines()) >= 5, "SHAP output too short"
