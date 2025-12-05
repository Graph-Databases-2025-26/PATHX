"""
Key-Scan Operator - Algorithm 2 from the paper
Two-step retrieval: keys first, then populate remaining attributes
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .base_scan import BaseScanOperator, _filter_new_rows
from ..prompts.templates import PromptTemplates
from ..utils.response_parser import ResponseParser


class KeyScanOperator(BaseScanOperator):
    """
    Key-Scan implementation (Algorithm 2 in GALOIS paper).
    
    Strategy:
    1. Retrieve all key values iteratively
    2. For each key, get remaining attributes in a single call
    3. This approach improves quality through focused prompts
    
    More expensive than Table-Scan but generally higher quality.
    Can be parallelized in the second step.
    """

    def __init__(self, llm, max_iters: int = 5, key_attributes: Optional[List[str]] = None):
        """
        Args:
            llm: LangChain ChatModel instance
            max_iters: Maximum iterations for key retrieval
            key_attributes: List of attribute names that form the key
                           If None, assumes first attribute is key
        """
        super().__init__(llm, max_iters)
        self.key_attributes = key_attributes

    def scan(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute Key-Scan with two-step retrieval.
        
        Step 1: Get all keys
        Step 2: For each key, get remaining attributes
        
        Returns:
            Dict with result_set, time, tokens, and metadata
        """
        total_time = 0.0
        total_tokens = 0

        # Determine key attributes
        key_attrs = self.key_attributes or [attributes[0]]  # Default: first attr
        non_key_attrs = [attr for attr in attributes if attr not in key_attrs]

        # Step 1: Retrieve all keys
        keys_data, keys_time, keys_tokens = self._retrieve_keys(
            dataset=dataset,
            table=table,
            key_attributes=key_attrs,
            where_clause=where_clause,
        )
        total_time += keys_time
        total_tokens += keys_tokens

        # Step 2: For each key, get remaining attributes
        all_rows: List[Dict] = []
        
        for key_row in keys_data:
            if not non_key_attrs:
                # No additional attributes needed
                all_rows.append(key_row)
                continue

            # Get remaining attributes for this key
            tuple_data, tuple_time, tuple_tokens = self._retrieve_tuple(
                dataset=dataset,
                table=table,
                non_key_attributes=non_key_attrs,
                key_row=key_row,
            )
            total_time += tuple_time
            total_tokens += tuple_tokens

            # Merge key and non-key attributes
            full_row = {**key_row, **tuple_data}
            all_rows.append(full_row)

        return {
            "result_set": all_rows,
            "time": round(total_time, 3),
            "tokens": total_tokens,
            "meta": {
                "num_keys": len(keys_data),
                "strategy": "key_scan",
            },
        }

    def _retrieve_keys(
        self,
        dataset: str,
        table: str,
        key_attributes: List[str],
        where_clause: Optional[str],
    ) -> tuple[List[Dict], float, int]:
        """
        Step 1: Iteratively retrieve all key values.
        Follow-up prompts include context of previously retrieved keys.
        """
        all_keys: List[Dict] = []
        seen: Set = set()

        total_time = 0.0
        total_tokens = 0

        # -------- FIRST ITERATION --------
        prompt = PromptTemplates.key_scan_first(
            dataset=dataset,
            table=table,
            key_attributes=key_attributes,
            where_clause=where_clause,
        )

        raw, elapsed, tokens = self._call_llm(prompt)
        total_time += elapsed
        total_tokens += tokens

        content = self._get_content(raw)
        parsed = ResponseParser.parse_llm_response(content, elapsed, tokens)

        keys_batch = parsed["result_set"]
        new_keys = _filter_new_rows(keys_batch, seen)
        all_keys.extend(new_keys)

        # If no keys found in first iteration then stop immediately
        if not new_keys:
            return all_keys, total_time, total_tokens

        # -------- FOLLOW-UP ITERATIONS --------
        for i in range(1, self.max_iters):
            followup_prompt = PromptTemplates.key_scan_followup(previous_keys=all_keys)

            raw, elapsed, tokens = self._call_llm(followup_prompt)
            total_time += elapsed
            total_tokens += tokens

            content = self._get_content(raw)
            parsed = ResponseParser.parse_llm_response(content, elapsed, tokens)

            keys_batch = parsed["result_set"]
            only_new = _filter_new_rows(keys_batch, seen)

            # If no new keys found then stop
            if not only_new:
                break

            all_keys.extend(only_new)

        return all_keys, total_time, total_tokens


    def _retrieve_tuple(
        self,
        dataset: str,
        table: str,
        non_key_attributes: List[str],
        key_row: Dict[str, Any],
    ) -> tuple[Dict[str, Any], float, int]:
        """
        Step 2: Get remaining attributes for a specific key.

        Returns:
            (dict_of_non_key_values, time, tokens)
        """
        # Extract key name and value (single-attribute key assumption)
        key_name = list(key_row.keys())[0]
        key_value = key_row[key_name]

        prompt = PromptTemplates.key_scan_tuple(
            dataset=dataset,
            table=table,
            non_key_attributes=non_key_attributes,
            key_name=key_name,
            key_value=key_value,
        )

        raw, elapsed, tokens = self._call_llm(prompt)
        content = self._get_content(raw)

        try:
            # Response should be flat dict: {"attr1": "val1", "attr2": "val2", ...}
            response_json = ResponseParser.extract_json(content)

            # Ensure the result contains all non_key_attributes
            tuple_data = {
                attr: response_json.get(attr, None)
                for attr in non_key_attributes
            }

            return tuple_data, elapsed, tokens

        except Exception as e:
            print(f"Warning: Failed to parse tuple data for key {key_name}={key_value}: {e}")
            print(f"  Raw content (truncated): {content[:200]}...")
            # On parse error, return None for all fields
            return {attr: None for attr in non_key_attributes}, elapsed, tokens
