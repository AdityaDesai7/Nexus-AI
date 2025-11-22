# trading_bot/llm/llm_wrapper.py
import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLM:
    """
    Unified LLM interface for all agents.
    Each agent uses:    self.llm.ask(prompt)
    Internally uses Groq's ChatCompletion API (object-based response).
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ERROR: GROQ_API_KEY not found in environment.")

        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")

        # Default model
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def ask(self, prompt: str) -> str:
        """
        Send prompt → Groq LLM → return final text.
        Always returns a SAFE string (never crashes the agents).
        """
        try:
            logger.info("LLM → Groq request started")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert multi-agent trading assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )

            # ❗ FIXED: Groq returns object, not dict
            message = response.choices[0].message
            answer = message.content.strip()

            logger.info("LLM → Groq request finished")
            return answer

        except Exception as e:
            logger.error(f"Groq LLM error: {e}")
            return f"[LLM_ERROR] {e}"
