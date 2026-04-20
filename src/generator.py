import requests
from .config import GENERATION_TEMPERATURE, OLLAMA_API_URL, LLM_MODEL_NAME, REQUEST_TIMEOUT


class AnswerGenerator:
    def __init__(self):
        """Initialize the generator with configuration values."""
        self.url = OLLAMA_API_URL
        self.model = LLM_MODEL_NAME

    def create_prompt(self, complaint: str, examples: list) -> str:
        """
        Build the final prompt from the user complaint and retrieved examples.
        """
        context_block = ""
        for i, ex in enumerate(examples):
            context_block += f"=== EXAMPLE {i + 1} ===\n"
            context_block += f"INPUT: {ex['complaint']}\n"
            context_block += f"OUTPUT: {ex['response']}\n\n"

        prompt = f"""
You are an intelligent assistant for a prosecutor's office employee.
Your task is to draft an official response to a citizen complaint using the provided examples.

STYLE: Strictly formal and businesslike. Do not use emotional language or first-person wording.
Write the final response in Russian.
Use standard prosecutor-office wording for reviewed complaints, established facts, explanations, and appeal rights.

CONTEXT (examples of correct responses):
{context_block}

TASK:
Write a response to the following complaint:
"{complaint}"

Your response:
        """
        return prompt

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to Ollama and return generated text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": GENERATION_TEMPERATURE
            }
        }

        try:
            print("Sending request to the language model...")
            response = requests.post(
                self.url,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "Error: empty model response")

        except requests.exceptions.ConnectionError:
            return "Error: could not connect to Ollama. Check that Ollama is running."
        except Exception as e:
            return f"Generation error: {e}"
