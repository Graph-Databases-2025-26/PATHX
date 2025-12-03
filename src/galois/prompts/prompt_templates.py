"""
Centralized prompt templates for all Galois operations
"""
from typing import List, Optional, Any
import json


class PromptTemplates:
    """All prompt templates used across Galois variants"""

    @staticmethod
    def table_scan_first(
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> str:
        """Initial Table-Scan prompt"""
        attrs_text = ", ".join(attributes)
        cond_text = (
            f"Only return rows that satisfy ALL of the following condition(s): {where_clause}"
            if where_clause
            else "Return any rows that are correct and consistent with the table."
        )
        
        return f"""You are an expert data assistant.

We have a table "{table}" in dataset "{dataset}".
We are interested in the following attributes (columns): {attrs_text}.

Task:
- Produce rows from this table.
- {cond_text}

Output requirements:
- Return ONLY valid JSON with NO explanations, NO markdown, NO extra text
- Use this exact format: {{"result_set": [{{"attr1": "val1", "attr2": "val2"}}, ...]}}
- Do NOT add comments, examples, or any text before/after the JSON
- If no rows match, return: {{"result_set": []}}

Start your response with {{ and end with }}"""

    @staticmethod
    def table_scan_followup() -> str:
        """Follow-up Table-Scan prompt for more rows"""
        return """List more values if there are more, otherwise return an empty result set.

Do NOT repeat any row that was already provided.

CRITICAL: Return ONLY valid JSON with NO explanations.
Format: {"result_set": [...]} or {"result_set": []}

Start your response with { and end with }"""

    @staticmethod
    def key_scan_first(
        dataset: str,
        table: str,
        key_attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> str:
        """Initial Key-Scan prompt to retrieve keys"""
        keys_text = ", ".join(key_attributes)
        cond_text = (
            f"where the following condition holds: {where_clause}"
            if where_clause
            else ""
        )
        
        return f"""You are an expert data assistant.

List the unique key values for table "{table}" in dataset "{dataset}".
Key attributes: {keys_text}
{cond_text if cond_text else "List all unique keys."}

CRITICAL: Return ONLY valid JSON with NO explanations, NO markdown.
Format: {{"result_set": [{{"key": "value1"}}, {{"key": "value2"}}, ...]}}

Start your response with {{ and end with }}"""

    @staticmethod
    def key_scan_followup() -> str:
        """Follow-up Key-Scan prompt for more keys"""
        return """List MORE unique key values if there are more.

Do NOT repeat previous values. Return empty result_set if no more keys exist.

Return ONLY valid JSON:
{"result_set": []}"""

    @staticmethod
    def key_scan_tuple(
        dataset: str,
        table: str,
        non_key_attributes: List[str],
        key_name: str,
        key_value: Any,
    ) -> str:
        """Prompt to get remaining attributes for a specific key"""
        attrs_text = ", ".join(non_key_attributes)
        expected_format = ", ".join(f'"{attr}": value' for attr in non_key_attributes)
        
        return f"""You are an expert data assistant.

For table "{table}" in dataset "{dataset}", provide the values of attributes: {attrs_text}
for the row where {key_name} = "{key_value}".

CRITICAL: Return ONLY valid JSON with NO explanations, NO markdown, NO extra text.
Format: {{{expected_format}}}

Example: {{"population": 12345, "gdp": 999999999}}

Start your response with {{ and end with }}"""

    @staticmethod
    def cell_value(
        dataset: str,
        table: str,
        attribute: str,
        key_name: str,
        key_value: Any,
    ) -> str:
        """Prompt to get a single cell value (Galois_WO)"""
        return f"""For table "{table}" in dataset "{dataset}",
what is the value of attribute "{attribute}" for the row with {key_name} = "{key_value}"?

CRITICAL: Return ONLY valid JSON with NO explanations, NO markdown blocks, NO extra text.
Format: {{"{attribute}": value}}

Example correct responses:
{{"{attribute}": 12345}}
{{"{attribute}": "text value"}}
{{"{attribute}": null}}

Do NOT include any text before or after the JSON object."""

    @staticmethod
    def evaluate_selectivity(
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> str:
        """Prompt for Galois_S to evaluate condition selectivity"""
        cond_list = "\n".join(f"  {i+1}. {cond}" for i, cond in enumerate(conditions))
        
        return f"""You are a database query optimizer.

For table "{table}" in dataset "{dataset}", evaluate the SELECTIVITY of each condition.
Selectivity = how many rows this condition will FILTER OUT (keep few rows = higher selectivity).

Conditions:
{cond_list}

For each condition, respond with:
- "higher" if it's very selective (filters out many rows, keeps few)
- "lower" if it's not selective (keeps many rows)

Return ONLY valid JSON mapping condition to selectivity:
{{
  "{conditions[0] if conditions else 'condition'}": "higher",
  ...
}}"""

    @staticmethod
    def evaluate_confidence(
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> str:
        """Prompt for Galois_F to evaluate confidence per condition"""
        cond_list = "\n".join(f"  {i+1}. {cond}" for i, cond in enumerate(conditions))
        
        return f"""You are evaluating your own knowledge confidence.

For table "{table}" in dataset "{dataset}", how confident are you in returning CORRECT data for each condition?

Conditions:
{cond_list}

For each condition, respond with:
- "high" if you are very confident about this data
- "low" if you are uncertain or this data is rare/obscure

Return ONLY valid JSON:
{{
  "{conditions[0] if conditions else 'condition'}": "high",
  ...
}}"""

    @staticmethod
    def evaluate_key_confidence(
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str],
    ) -> str:
        """Prompt for Galois_F to estimate confidence in key retrieval"""
        attrs_text = ", ".join(attributes)
        cond_text = f" WHERE {where_clause}" if where_clause else ""
        
        return f"""Evaluate your confidence in retrieving COMPLETE and CORRECT key values.

Query context:
- Dataset: {dataset}
- Table: {table}
- Attributes: {attrs_text}
- Conditions: {cond_text if cond_text else "none"}

On a scale from 0.0 (no confidence) to 1.0 (complete confidence), 
how confident are you that you can retrieve ALL the correct key values?

Consider:
- How common/rare is this data in your training?
- How complex are the conditions?
- How certain are you about this domain?

Return ONLY valid JSON:
{{"confidence": 0.25}}"""


# Convenience functions
def build_table_scan_first(dataset: str, table: str, attributes: List[str], 
                           where_clause: Optional[str] = None) -> str:
    return PromptTemplates.table_scan_first(dataset, table, attributes, where_clause)


def build_table_scan_followup() -> str:
    return PromptTemplates.table_scan_followup()


def build_key_scan_first(dataset: str, table: str, key_attributes: List[str],
                         where_clause: Optional[str] = None) -> str:
    return PromptTemplates.key_scan_first(dataset, table, key_attributes, where_clause)


def build_key_scan_followup() -> str:
    return PromptTemplates.key_scan_followup()