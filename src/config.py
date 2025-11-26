from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field

load_dotenv()


class WatsonXConfig(BaseModel):
    api_key: str
    endpoint: str
    project_id: str
    model: str
    temperature: float
    max_tokens: int


class GeminiConfig(BaseModel):
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int


class AzureConfig(BaseModel):
    api_key: str
    endpoint: str
    deployment: str
    api_version: str
    model: str
    temperature: float
    max_tokens: int


class DatabaseConfig(BaseModel):
    type: str
    db_dir: str
    data_dir: str
    ingest_on_demand: bool


class IOConfig(BaseModel):
    sql_output: str
    nl_output: str
    prompts_dir: str


class IterativeConfig(BaseModel):
    max_iters: int
    stop_if_no_new: bool


class LoggingConfig(BaseModel):
    level: str
    json: bool


class EvaluationConfig(BaseModel):
    enable: bool
    metrics: list[str]
    output_format: str
    aggregate: bool


class AppConfig(BaseModel):
    llm_provider: str
    prompt_strategy: str

    watsonx: WatsonXConfig
    gemini: GeminiConfig
    azure: AzureConfig

    database: DatabaseConfig
    io: IOConfig
    iterative: IterativeConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig

    @classmethod
    def load(cls, path: str = "config/AppConfig.yaml") -> "AppConfig":
        """
        Load YAML, substitute ${ENV_VARS} and validate with Pydantic
        """
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        raw = os.path.expandvars(raw)
        data = yaml.safe_load(raw)
        return cls(**data)


def load_config() -> AppConfig:
    return AppConfig.load()
