# embedding/qa_engine.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

class QAEngine:
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model = model_name
        self._verify_connection()

    def _verify_connection(self):
        try:
            requests.get("http://localhost:11434", timeout=3)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then in a new terminal: ollama pull llama3.2:1b"
            )

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are an expert technical assistant helping a repair technician.

You have been given the following excerpts from a technical manual:

{context}

Based ONLY on the information above, answer the following question in a professional and helpful way.
- Synthesise the information — do not copy sentences verbatim.
- Structure your answer clearly: start with a direct answer, then explain the reasoning or steps.
- If the context does not contain enough information, say so clearly.
- Use precise technical language appropriate for a repair professional.

Question: {question}

Answer:"""

    def answer_question(self, question: str, context: str) -> str:
        # Guard: if context is empty for any reason, return early
        if not context or not context.strip():
            return "No relevant context was retrieved to answer this question."

        prompt = self._build_prompt(question, context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "num_predict": 300,
                "num_ctx": 1024, 
            }
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except requests.exceptions.Timeout:
            return "Answer generation timed out. Try a shorter query or restart Ollama."
        except Exception as e:
            return f"Answer generation failed: {str(e)}"