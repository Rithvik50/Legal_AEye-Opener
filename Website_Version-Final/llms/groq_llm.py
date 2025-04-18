from typing import List, Optional
from langchain_core.language_models import LLM
from groq import Groq

class GroqLLM(LLM):
    model: str = "llama3-70b-8192"
    groq_api_key: str = "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"
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = Groq(api_key=self.groq_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    @property
    def _llm_type(self) -> str:
        return "groq"
