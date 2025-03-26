# Tests if Ollama is running and agent can return a basic response

from agent.shap_agent import explain_with_agent, check_ollama_alive

def test_ollama_connection():
    assert check_ollama_alive(), "Ollama server is not running"

def test_agent_response_format():
    summary = "- mean radius: 0.12\n- mean texture: 0.11"
    explanation = explain_with_agent(summary, (100, 10))
    assert isinstance(explanation, str), "Agent response must be a string"
    assert len(explanation.strip()) > 10, "Agent response is too short"
