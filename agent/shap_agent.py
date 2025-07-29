# agent/shap_agent.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class ShapAgent:
    def __init__(self):
        self.provider = os.getenv("AGENT_PROVIDER", "ollama")
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    @staticmethod
    def check_ollama_alive(timeout=5):
        """Check if Ollama is running with basic connectivity test"""
        try:
            response = requests.get("http://localhost:11434", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    def generate_explanation(self, summary_text, data_shape):
        """Generate explanation using the approach that previously worked"""
        if self.provider != "ollama":
            return "⚠️ Agent provider not supported. Check your .env file"

        prompt = f"""
You are an explicable AI (XAI) agent that explains how machine learning models work.
This dataset has {data_shape[0]} rows and {data_shape[1]} features.

Given the SHAP global summary below, explain in English what the model is focusing on and why.

SHAP Summary:
{summary_text}
"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "⚠️ Error: No response from Ollama.")
            else:
                return f"⚠️ Ollama API error: {response.text}"

        except requests.exceptions.Timeout:
            return (
                "⚠️ Ollama timeout - The model might still be loading.\n"
                f"Try: `ollama pull {self.model}` and wait a few minutes\n"
                "Large models can take several minutes to load initially."
            )
        except requests.exceptions.RequestException as e:
            return f"⚠️ Connection error: {str(e)}\nMake sure Ollama is running: `ollama serve`"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"

    @staticmethod
    def explain_with_agent(summary_text: str, data_shape: tuple) -> str:
        agent = ShapAgent()
        return agent.generate_explanation(summary_text, data_shape)
    
    @staticmethod
    def use_ollama_agent(summary_text: str, data_shape: tuple) -> str:
        agent = ShapAgent()
        return agent.generate_explanation(summary_text, data_shape)
