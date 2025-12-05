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
        if where_clause:
            cond_text = (
                "Prefer rows that satisfy ALL of the following condition(s): "
                f"{where_clause}. "
                "If you are not completely sure, you can still return plausible rows that approximately match."
            )
        else:
            cond_text = "Return any rows that are correct and consistent with the table."

        return f"""You are an expert data assistant.

We have a table "{table}" in dataset "{dataset}".
We are interested in the following attributes (columns): {attrs_text}.

Task:
- Produce multiple rows from this table.
- {cond_text}

Output requirements:
- Return ONLY valid JSON with NO explanations, NO markdown, NO extra text
- Use this exact format: {{"result_set": [{{"attr1": "val1", "attr2": "val2"}}, ...]}}
- Do NOT add comments, examples, or any text before/after the JSON
- If you believe no rows match at all, return: {{"result_set": []}}

Start your response with {{ and end with }}"""


    @staticmethod
    def table_scan_followup(previous_rows: List[dict]) -> str:
        """Follow-up Table-Scan prompt for more rows, with explicit context."""
        # Possible to limit to last N rows for security/performance
        preview_rows = previous_rows[-50:]
        rows_json = json.dumps(preview_rows, ensure_ascii=False, indent=2)

        return f"""You are an expert data assistant.

You previously provided the following rows (do NOT repeat them):

{rows_json}

Now list MORE rows from the same table that satisfy the same conditions.
If there are no more rows, return an empty result set.

CRITICAL: Return ONLY valid JSON with NO explanations.
Format: {{"result_set": [{{"attr1": "val1", "attr2": "val2"}}, ...]}} or {{"result_set": []}}

Start your response with {{ and end with }}"""

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

        # Используем реальное имя ключевого атрибута в примере
        example_key = key_attributes[0] if key_attributes else "key"
        
        return f"""You are an expert data assistant.

List the unique key values for table "{table}" in dataset "{dataset}".
Key attributes: {keys_text}
{cond_text if cond_text else "List all unique keys."}

CRITICAL: Return ONLY valid JSON with NO explanations, NO markdown.
Format: {{"result_set": [{{"{example_key}": "value1"}}, {{"{example_key}": "value2"}}, ...]}}

Start your response with {{ and end with }}"""


    @staticmethod
    def key_scan_followup(previous_keys: List[dict]) -> str:
        """Follow-up Key-Scan prompt for more keys, with explicit context."""
        preview_keys = previous_keys[-100:]
        keys_json = json.dumps(preview_keys, ensure_ascii=False, indent=2)

        return f"""You are an expert data assistant.

You previously listed the following key values (do NOT repeat them):

{keys_json}

List MORE unique key values for the same table, if there are any.
If there are no more keys, return an empty result set.

Return ONLY valid JSON with NO explanations or markdown.
Format: {{"result_set": [{{"key_attr": "value"}}, ...]}} or {{"result_set": []}}

Start your response with {{ and end with }}"""


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
        return f"""You are an expert data assistant with strong knowledge of real-world data.

We have a table "{table}" in dataset "{dataset}".
You are given the key of a specific row: {key_name} = "{key_value}".

Your task:
- Return the value of attribute "{attribute}" for THIS row.
- Use your best estimate based on real-world information.
- Prefer an approximate but reasonable value over null.

CRITICAL OUTPUT RULES:
- Return ONLY valid JSON with NO explanations, NO markdown blocks, NO extra text.
- If the attribute is NUMERIC (like population, area, counts, GDP, etc.):
  - Return a bare number (integer) WITHOUT quotes, units, or words.
  - Example: {{"{attribute}": 3769000}}
- If the attribute is TEXT (like country name, capital city, category):
  - Return a quoted string.
  - Example: {{"{attribute}": "Germany"}}
- Only return null if the value is genuinely not applicable (very rare).

Valid examples:
{{"{attribute}": 3769000}}
{{"{attribute}": 1710000}}
{{"{attribute}": 25}}
{{"{attribute}": "Germany"}}

INVALID examples (do NOT do this):
{{"{attribute}": "about 3.7 million"}}
{{"{attribute}": "3.7M people"}}
{{"{attribute}": "663,267 square kilometers"}}
{{"{attribute}": null}}  # avoid null unless truly impossible to estimate

Format to follow:
{{"{attribute}": <value>}}

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