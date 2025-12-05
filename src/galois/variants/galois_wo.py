"""
Galois_WO: Without Optimizations
Baseline approach that retrieves data cell-by-cell (most expensive)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..operators.base_scan import BaseScanOperator, _filter_new_rows
from ..prompts.prompt_templates import PromptTemplates
from ..utils.response_parser import ResponseParser


class GaloisWOOperator(BaseScanOperator):
    """
    Galois_WO: The most expensive baseline.
    
    Strategy:
    1. Retrieve all keys (no pushdown)
    2. For each key:
       3. For EACH attribute:
          4. Make separate LLM call to get that single cell
    
    This demonstrates the worst case and shows why optimizations matter.
    Consumes 8.5M-19.7M tokens vs 0.95M-1.72M for optimized versions.
    """

    def __init__(self, llm, max_iters: int = 5, key_attributes: Optional[List[str]] = None):
        """
        Args:
            llm: LangChain ChatModel instance
            max_iters: Maximum iterations for key retrieval
            key_attributes: List of key attribute names (default: first attr)
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
        Execute Galois_WO cell-by-cell scan.
        
        Note: where_clause is NOT pushed down to key retrieval,
        it's only applied as post-filter.
        """
        total_time = 0.0
        total_tokens = 0

        # Determine keys
        key_attrs = self.key_attributes or [attributes[0]]
        non_key_attrs = [attr for attr in attributes if attr not in key_attrs]

        # Step 1: Get ALL keys (no condition pushdown)
        keys_data, keys_time, keys_tokens = self._retrieve_keys(
            dataset=dataset,
            table=table,
            key_attributes=key_attrs,
            # NOTE: We deliberately don't pass where_clause here
        )
        total_time += keys_time
        total_tokens += keys_tokens

        # Step 2: For each key, populate CELL BY CELL
        all_rows: List[Dict] = []
        
        for key_row in keys_data:
            full_row = dict(key_row)  # Start with key
            
            # Get each cell individually
            for attr in non_key_attrs:
                cell_value, cell_time, cell_tokens = self._get_cell_value(
                    dataset=dataset,
                    table=table,
                    attribute=attr,
                    key_row=key_row,
                )
                total_time += cell_time
                total_tokens += cell_tokens
                full_row[attr] = cell_value
            
            all_rows.append(full_row)

        # DEBUG: raw data before WHERE filter
        if where_clause:
            print("\n[DEBUG][Galois_WO] Rows BEFORE WHERE filter "
                  f"({len(all_rows)} rows):")
            for r in all_rows[:5]:   # just first 5 rows for brevity
                print("  ", r)


        # Step 3: Apply WHERE clause as post-filter
        if where_clause:
            all_rows = self._filter_rows(all_rows, where_clause)

        return {
            "result_set": all_rows,
            "time": round(total_time, 3),
            "tokens": total_tokens,
            "meta": {
                "num_keys": len(keys_data),
                "strategy": "cell_by_cell",
                "num_cell_calls": len(keys_data) * len(non_key_attrs),
            },
        }

    def _retrieve_keys(
        self,
        dataset: str,
        table: str,
        key_attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> tuple[List[Dict], float, int]:
        """Retrieve all key values (similar to Key-Scan step 1).

        NOTE: For Galois_WO we deliberately do NOT push down the WHERE clause,
        so where_clause is expected to be None and is not passed in normal usage.
        """
        all_keys: List[Dict] = []
        seen: Set = set()
        
        total_time = 0.0
        total_tokens = 0

        # -------- FIRST ITERATION (no pushdown) --------
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

        if new_keys:
            all_keys.extend(new_keys)

        # If no new keys found in first iteration then stop
        if not new_keys:
            return all_keys, total_time, total_tokens

        # -------- FOLLOW-UP ITERATIONS with context --------
        for i in range(1, self.max_iters):
            followup_prompt = PromptTemplates.key_scan_followup(previous_keys=all_keys)

            raw, elapsed, tokens = self._call_llm(followup_prompt)
            total_time += elapsed
            total_tokens += tokens

            content = self._get_content(raw)
            parsed = ResponseParser.parse_llm_response(content, elapsed, tokens)
            
            keys_batch = parsed["result_set"]
            only_new = _filter_new_rows(keys_batch, seen)

            if not only_new:
                break

            all_keys.extend(only_new)

        return all_keys, total_time, total_tokens

    def _get_cell_value(
        self,
        dataset: str,
        table: str,
        attribute: str,
        key_row: Dict[str, Any],
    ) -> tuple[Any, float, int]:
        """
        Get a SINGLE cell value (the expensive part).
        
        Returns:
            (cell_value, time, tokens)
        """
        key_name = list(key_row.keys())[0]
        key_value = key_row[key_name]

        prompt = PromptTemplates.cell_value(
            dataset=dataset,
            table=table,
            attribute=attribute,
            key_name=key_name,
            key_value=key_value,
        )

        raw, elapsed, tokens = self._call_llm(prompt)
        content = self._get_content(raw)

        # DEBUG: see what the response was
        # print(f"\n[DEBUG][Galois_WO] cell_value for {key_name}={key_value}, "
        #      f"attr={attribute}:")
        # print(content[:300], "...\n")

        try:
            response_json = ResponseParser.extract_json(content)
            cell_value = response_json.get(attribute)
            return cell_value, elapsed, tokens
        except Exception as e:
            # Log warning but continue (return None for this cell)
            print(f"Warning: Failed to get cell value for {attribute}: {e}")
            print(f"  Response was: {content[:200]}...")
            return None, elapsed, tokens

    def _filter_rows(self, rows: List[Dict], where_clause: str) -> List[Dict]:
        """
        Apply WHERE clause filtering in memory.
        
        Uses WhereEvaluator for complete SQL WHERE evaluation.
        """
        from ..utils.where_evaluator import WhereEvaluator
        
        print(f"Applying WHERE clause post-filter: {where_clause}")
        return WhereEvaluator.filter_rows(rows, where_clause)