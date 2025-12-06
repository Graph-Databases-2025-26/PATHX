"""
Centralized prompt templates for all Galois operations, rendered via Jinja2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import json

from jinja2 import Environment, FileSystemLoader

# Directory with .j2 files: src/galois/prompts/jinja/*.j2
_TEMPLATES_DIR = Path(__file__).resolve().parent / "jinja"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


class PromptTemplates:
    """All prompt templates used across Galois variants (Jinja2-based)."""

    @staticmethod
    def _render(template_name: str, **context: Any) -> str:
        """Internal helper: render template_name.j2 with given context."""
        template = _env.get_template(f"{template_name}.j2")
        return template.render(**context)

    # ------------------------------------------------------------------
    # Table scan
    # ------------------------------------------------------------------
    @staticmethod
    def table_scan_first(
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> str:
        attrs_text = ", ".join(attributes)
        return PromptTemplates._render(
            "table_scan_first",
            dataset=dataset,
            table=table,
            attributes=attributes,
            attrs_text=attrs_text,
            where_clause=where_clause,
        )

    @staticmethod
    def table_scan_followup(previous_rows: List[dict]) -> str:
        # limit context size a bit
        preview_rows = previous_rows[-50:]
        rows_json = json.dumps(preview_rows, ensure_ascii=False, indent=2)
        return PromptTemplates._render(
            "table_scan_followup",
            previous_rows=preview_rows,
            rows_json=rows_json,
        )

    # ------------------------------------------------------------------
    # Key scan
    # ------------------------------------------------------------------
    @staticmethod
    def key_scan_first(
        dataset: str,
        table: str,
        key_attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> str:
        keys_text = ", ".join(key_attributes)
        example_key = key_attributes[0] if key_attributes else "key"
        return PromptTemplates._render(
            "key_scan_first",
            dataset=dataset,
            table=table,
            key_attributes=key_attributes,
            keys_text=keys_text,
            where_clause=where_clause,
            example_key=example_key,
        )

    @staticmethod
    def key_scan_followup(previous_keys: List[dict]) -> str:
        preview_keys = previous_keys[-100:]
        keys_json = json.dumps(preview_keys, ensure_ascii=False, indent=2)
        return PromptTemplates._render(
            "key_scan_followup",
            previous_keys=preview_keys,
            keys_json=keys_json,
        )

    @staticmethod
    def key_scan_tuple(
        dataset: str,
        table: str,
        non_key_attributes: List[str],
        key_name: str,
        key_value: Any,
    ) -> str:
        attrs_text = ", ".join(non_key_attributes)
        expected_format = ", ".join(f'"{attr}": value' for attr in non_key_attributes)
        return PromptTemplates._render(
            "key_scan_tuple",
            dataset=dataset,
            table=table,
            non_key_attributes=non_key_attributes,
            attrs_text=attrs_text,
            key_name=key_name,
            key_value=key_value,
            expected_format=expected_format,
        )

    # ------------------------------------------------------------------
    # Cell value (WO)
    # ------------------------------------------------------------------
    @staticmethod
    def cell_value(
        dataset: str,
        table: str,
        attribute: str,
        key_name: str,
        key_value: Any,
    ) -> str:
        return PromptTemplates._render(
            "cell_value",
            dataset=dataset,
            table=table,
            attribute=attribute,
            key_name=key_name,
            key_value=key_value,
        )

    # ------------------------------------------------------------------
    # Selectivity / confidence (S / F)
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_selectivity(
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> str:
        return PromptTemplates._render(
            "evaluate_selectivity",
            dataset=dataset,
            table=table,
            conditions=conditions,
        )

    @staticmethod
    def evaluate_confidence(
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> str:
        return PromptTemplates._render(
            "evaluate_confidence",
            dataset=dataset,
            table=table,
            conditions=conditions,
        )

    @staticmethod
    def evaluate_key_confidence(
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str],
    ) -> str:
        attrs_text = ", ".join(attributes)
        return PromptTemplates._render(
            "evaluate_key_confidence",
            dataset=dataset,
            table=table,
            attributes=attributes,
            attrs_text=attrs_text,
            where_clause=where_clause,
        )


# ----------------------------------------------------------------------
# Optional convenience functions (fixed signatures)
# ----------------------------------------------------------------------
def build_table_scan_first(
    dataset: str,
    table: str,
    attributes: List[str],
    where_clause: Optional[str] = None,
) -> str:
    return PromptTemplates.table_scan_first(dataset, table, attributes, where_clause)


def build_table_scan_followup(previous_rows: List[dict]) -> str:
    return PromptTemplates.table_scan_followup(previous_rows)


def build_key_scan_first(
    dataset: str,
    table: str,
    key_attributes: List[str],
    where_clause: Optional[str] = None,
) -> str:
    return PromptTemplates.key_scan_first(dataset, table, key_attributes, where_clause)


def build_key_scan_followup(previous_keys: List[dict]) -> str:
    return PromptTemplates.key_scan_followup(previous_keys)
