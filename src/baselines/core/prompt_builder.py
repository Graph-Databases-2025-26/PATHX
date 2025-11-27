from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .response_model import PARSER

PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_ROOT)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render(template_rel_path: str, context: Dict[str, Any]) -> str:
    tmpl = _env.get_template(template_rel_path)
    return tmpl.render(**context).strip()


def build_sql_prompt(
    strategy: str,
    dataset: str,
    sql: str,
    schema_info: str,
) -> str:
    format_instructions = PARSER.get_format_instructions()
    template = f"sql/{strategy}.j2"
    ctx = {
        "dataset": dataset,
        "sql": sql,
        "schema_info": schema_info,
        "format_instructions": format_instructions,
    }
    return _render(template, ctx)


def build_nl_prompt(
    strategy: str,
    dataset: str,
    nl_question: str,
    schema_info: str,
) -> str:
    format_instructions = PARSER.get_format_instructions()
    template = f"nl/{strategy}.j2"
    ctx = {
        "dataset": dataset,
        "nl_question": nl_question,
        "schema_info": schema_info,
        "format_instructions": format_instructions,
    }
    return _render(template, ctx)


def build_followup_prompt(
    dataset: str,
    previous_rows: List[Dict[str, Any]],
    nl_question: str | None = None,
    sql: str | None = None,
) -> str:
    format_instructions = PARSER.get_format_instructions()
    template = "common/followup.j2"

    previous_rows_json = json.dumps(previous_rows, ensure_ascii=False, indent=2)

    ctx = {
        "dataset": dataset,
        "nl_question": nl_question,
        "sql": sql,
        "previous_rows_json": previous_rows_json,
        "format_instructions": format_instructions,
    }
    return _render(template, ctx)
