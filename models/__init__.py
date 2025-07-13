import os

def get_model_names():
    """Return list of available model filenames"""
    return [
        f[:-3] for f in os.listdir(os.path.dirname(__file__))
        if f.endswith(".py") and f != "__init__.py"
    ]