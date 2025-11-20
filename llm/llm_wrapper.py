# llm_wrapper.py — FINAL ERROR-FREE VERSION
import os
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------
# Fallback LLM (dummy)
# ------------------------------------------
class DummyLLM:
    """Fallback deterministic LLM stub for offline/local testing."""
    def __init__(self):
        pass

    def ask(self, prompt: str) -> str:
        logger.info("DummyLLM used. Returning conservative answer.")
        return "DUMMY_RESPONSE: No LLM available."


# ------------------------------------------
# Try importing Groq
# ------------------------------------------
try:
    from groq import Groq

    class GroqLLM:
        """Real Groq LLM wrapper. Must expose .ask(prompt)."""
        def __init__(self, api_key=None, model_name=None):
            api_key = api_key or os.getenv("GROQ_API_KEY")
            model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

            if not api_key:
                raise ValueError("GROQ_API_KEY missing")

            self.client = Groq(api_key=api_key)
            self.model = model_name

        def ask(self, prompt: str) -> str:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": prompt}
                    ]
                )
                return resp.choices[0].message["content"]
            except Exception as e:
                logger.error(f"GroqLLM ask() failed: {e}")
                return f"[Groq error] {str(e)}"

    # IMPORTANT → expose CLASS, NOT INSTANCE
    LLM = GroqLLM

except Exception:
    # Groq not available → fallback
    LLM = DummyLLM
