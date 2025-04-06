# agent/shap_agent.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class ShapAgent:
    def __init__(self):
        self.provider = os.getenv("AGENT_PROVIDER", "ollama")
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # Default 60 segundos

    @staticmethod
    def check_ollama_alive(timeout=5):
        """Versión mejorada del check con timeout configurable"""
        try:
            res = requests.get("http://localhost:11434", timeout=timeout)
            return res.status_code == 200
        except requests.exceptions.RequestException:
            return False
        except Exception:
            return False

    def generate_explanation(self, summary_text, data_shape):
        """Combina lo mejor de ambas versiones con mejor manejo de errores"""
        if self.provider != "ollama":
            return "⚠️ Agent provider not supported. Check your .env file"

        prompt = self._build_prompt(summary_text, data_shape)
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "⚠️ No response from LLM")
            
        except requests.exceptions.Timeout:
            return "⚠️ Ollama timeout - Is the model loaded? Try: ollama pull " + self.model
        except requests.exceptions.RequestException as e:
            return f"⚠️ Ollama connection error: {str(e)}"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"

    def _build_prompt(self, summary_text, data_shape):
        """Construye el prompt manteniendo tu formato original"""
        return f"""
You are an AI assistant that explains how machine learning models work.
This dataset has {data_shape[0]} rows and {data_shape[1]} features.

Given the SHAP global summary below, explain in simple English what the model is focusing on and why.

SHAP Summary:
{summary_text}
"""