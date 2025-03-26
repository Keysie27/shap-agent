# 🤖 SHAP-Agent

**SHAP-Agent** is an explainability tool that combines [SHAP](https://github.com/slundberg/shap) values with a language model API to generate clear, human-friendly explanations of machine learning models.

This app is designed to:
- Analyze trained ML models using SHAP (global explanation)
- Convert SHAP summaries into natural language
- Allow users to upload their own models and data
- Include ready-to-use sample models and datasets for testing

## 🗂️ Project Structure

shap-agent/
│
├── agent/
│   └── shap_agent.py                 # GPT-based explanation logic
│
├── app/
│   ├── __init__.py                  # Marks app as a Python package
│   ├── main.py                      # Streamlit interface (UI entrypoint)
│   └── file_handler.py              # Handles uploaded files (models & CSV)
│
├── data/
│   ├── input_data/                  # User-uploaded datasets (.csv)
│   └── sample_data/                # Sample datasets for testing the app
│       ├── logistic_regression.csv
│       ├── random_forest.csv
│       └── xgboost.csv
│
├── input_models/                   # User-uploaded models (.pkl or .joblib)
│
├── sample_models/                  # Pretrained models for demo/testing
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── tests/
│   ├── test_agent.py               # Unit tests for GPT logic
│   ├── test_file_handler.py        # Tests for file loading
│   └── test_shap_explainer.py      # Tests for SHAP explanations
│
├── utils/
│   └── shap_explainer.py           # SHAP logic (creates summaries per model type)
│
├── .env                            # OpenAI API key and other env variables
├── .gitignore                      # Ignore venv, .env, models, data, etc.
├── README.md                       # Full documentation for using and running the app
└── requirements.txt                # Python dependencies

## 🚀 How to Use

### 🔧 Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/keysie27/shap-agent.git
cd shap-agent
```

2. **Create a virtual environment (optional)**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Install and start Ollama If you want to use a free, local agent:

Download and install Ollama from https://ollama.com/download

Then, run: 
```bash
ollama run mistral
```

4. **Environment Setup**

#### `.env` File
Create a `.env` file in the root of the project. This file stores secrets and api configuration values required. 
**Note:** An example `.env` file is available in the `tests/` directory to guide you. Make sure you set all the required variables.

5. **Run the app**
```bash
PYTHONPATH=. streamlit run app/main.py
```

## 🧪 Sample Usage
Use the models and datasets included in:

models/sample_models/
data/sample_data/

These are fully compatible with SHAP and can be used to test the app without uploading anything!

## 📥 For End Users

Users can:

- Upload their own models (.pkl or .joblib) to input_models/
- Upload matching datasets (.csv) to data/input_data/
- The app will run SHAP explainability and provide a GPT-based natural language explanation of model behavior.

## 💡 Notes on Agents
By default, this project uses Ollama with Mistral, which is:

- Free
- Runs locally on your machine (no API costs)
- Works offline

You can change to other agents later by modifying AGENT_PROVIDER in your .env.

## 🙌 Credits
- SHAP by Scott Lundberg et al.
- Ollama & Mistral by community contributors
- Streamlit
