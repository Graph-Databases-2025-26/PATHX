from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import yaml
from pydantic import BaseModel
from .config import load_config, AppConfig



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
    as_json: bool


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
        Loads YAML, put ${ENV_VARS}, validates via Pydantic.
        """
        path_obj = Path(path)
        raw = path_obj.read_text(encoding="utf-8")
        # ${VAR} → value from /.env
        raw = os.path.expandvars(raw)
        data = yaml.safe_load(raw)
        return cls(**data)


def load_config(path: str = "config/AppConfig.yaml") -> AppConfig:
    """
    Wrapper load_config().
    """
    return AppConfig.load(path)
