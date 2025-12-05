from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
import json
from src.baselines.core.response_model import PARSER

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_ROOT)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)

def _render(template_name: str, context: dict) -> str:
    """
    Renders a Jinja2 template with the given context.
    """
    tmpl = _env.get_template(template_name)
    return tmpl.render(**context).strip()

def build_llmscan_first(dataset: str, table: str, attributes, where_clause=None):
    """
    Builds a prompt for the first LLM scan request.
    """
    format_instructions = PARSER.get_format_instructions()

    # Build condition text
    cond_text = (
        f"Only return rows that satisfy ALL of the following condition(s): {where_clause}"
        if where_clause else
        "Return any rows that are correct and consistent with the table."
    )

    attrs_text = ", ".join(attributes) if attributes else "all relevant columns"

    # Build context dictionary
    ctx = {
        "dataset": dataset,
        "table": table,
        "attributes_text": attrs_text,
        "condition_text": cond_text,
        "format_instructions": format_instructions,
    }

    return _render("llm_scan_first.j2", ctx)


def build_llmscan_followup(dataset: str, table: str, attributes, previous_rows, where_clause=None):
    format_instructions = PARSER.get_format_instructions()
    """
    Builds a prompt for follow-up LLM scan requests.
    """

    # Build condition text
    cond_text = (
        f"Only return rows that satisfy ALL of the following condition(s): {where_clause}"
        if where_clause else
        "Return any additional rows that are correct and consistent with the table."
    )

    attrs_text = ", ".join(attributes) if attributes else "all relevant columns"

    # Build context dictionary
    ctx = {
        "dataset": dataset,
        "table": table,
        "attributes_text": attrs_text,
        "previous_rows_json": json.dumps(previous_rows, ensure_ascii=False, indent=2),
        "condition_text": cond_text,
        "format_instructions": format_instructions,
    }

    return _render("llm_scan_followup.j2", ctx)
