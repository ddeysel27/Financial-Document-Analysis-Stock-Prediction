import subprocess
import json

def ask_llm(prompt: str) -> str:
    """
    Calls a local LLaMA model using Ollama.
    Works offline. Free. No API keys needed.
    """
    try:
        # Run ollama with JSON output
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],  # You can change "llama3.2" → any model
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return f"LLM Error: {result.stderr}"

        return result.stdout.strip()

    except Exception as e:
        return f"⚠️ Ollama Runtime Error: {str(e)}"
