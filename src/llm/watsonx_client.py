import os
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model: str = "watsonx-llm"
    temperature: float = 0.0

class WatsonXClient:
    """
    Thin wrapper over WatsonX (stubbed). Replace `call()` with real SDK usage.
    """
    def __init__(self, api_key: str | None = None, cfg: LLMConfig = LLMConfig()):
        self.api_key = api_key or os.getenv("WATSONX_API_KEY", "")
        self.cfg = cfg
        if not self.api_key:
            raise RuntimeError("WATSONX_API_KEY is missing. Put it in config/.env")

    def call(self, prompt: str) -> dict:
        # TODO: integrate ibm-watsonx-ai SDK; return {"text": "...", "usage": {"tokens_in": x, "tokens_out": y}}
        # Temporary stub so pipeline runs:
        return {"text": '[]', "usage": {"tokens_in": 0, "tokens_out": 0}}
