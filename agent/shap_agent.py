# This file handles interaction with a local LLM (Ollama) to convert SHAP summaries into plain language

import os
import requests
from dotenv import load_dotenv

load_dotenv()

AGENT_PROVIDER = os.getenv("AGENT_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def explain_with_agent(summary_text: str, data_shape: tuple) -> str:
    """
    Main entry point to call the language agent.
    Currently supports only Ollama locally.
    """
    if AGENT_PROVIDER == "ollama":
        return use_ollama_agent(summary_text, data_shape)
    else:
        return "⚠️ Agent provider not supported. Please check your .env file."


def use_ollama_agent(summary_text: str, data_shape: tuple) -> str:
    """
    Uses Ollama API locally to generate explanation from SHAP summary.
    """
    prompt = f"""
You are an AI assistant that explains how machine learning models work.
This dataset has {data_shape[0]} rows and {data_shape[1]} features.

Given the SHAP global summary below, explain in simple English what the model is focusing on and why.

SHAP Summary:
{summary_text}
    """

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        result = response.json()
        return result.get("response", "⚠️ Error: No response from Ollama.")
    except Exception as e:
        return f"⚠️ Could not connect to Ollama. Make sure it is running. Error: {str(e)}"


# Check if Ollama is running
def check_ollama_alive():
    try:
        res = requests.get("http://localhost:11434")
        return res.status_code == 200
    except Exception:
        return False