"""
Base Scan Operator - Foundation for all Galois scan strategies
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage

Row = Dict[str, Any]
Table = List[Row]

def project_rows(rows: Table, attributes: List[str]) -> Table:
    """Project rows to requested attributes, preserving missing as None."""
    projected: Table = []
    attr_set = list(attributes)
    for r in rows:
        if not isinstance(r, dict):
            continue
        projected.append({a: r.get(a, None) for a in attr_set})
    return projected

def _make_hashable(value: Any) -> Any:
    """Convert any JSON-like value into a hashable form"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _make_hashable(v)) for k, v in sorted(value.items()))
    return repr(value)


def _row_signature(row: Row) -> Tuple:
    """Stable, hashable signature for a single row"""
    items = [(k, _make_hashable(v)) for k, v in row.items()]
    return tuple(sorted(items))


def _filter_new_rows(rows: Table, seen: Set[Tuple]) -> Table:
    """Keep only rows that are not in 'seen', update 'seen' in-place"""
    new_rows: Table = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sig = _row_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        new_rows.append(row)
    return new_rows


def extract_total_tokens(raw_response: BaseMessage) -> int:
    """
    Extract total tokens (input + output) for a single LLM call.
    Works for watsonx, Azure/OpenAI-style usage and Gemini (usage_metadata).
    """
    usage_meta = getattr(raw_response, "usage_metadata", None)
    if usage_meta:
        inp = usage_meta.get("input_tokens", 0) or 0
        out = usage_meta.get("output_tokens", 0) or 0
        return int(inp) + int(out)

    usage = getattr(raw_response, "usage", None)
    if usage is not None:
        if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens"):
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0
        elif isinstance(usage, dict):
            inp = usage.get("prompt_tokens", 0) or 0
            out = usage.get("completion_tokens", 0) or 0
        else:
            inp = out = 0
        return int(inp) + int(out)

    return 0


class BaseScanOperator(ABC):
    """
    Base class for all scan operators in Galois.
    
    Provides common functionality:
    - LLM invocation with timing and token tracking
    - Row deduplication
    - Basic prompt chain setup
    """

    def __init__(self, llm, max_iters: int = 5):
        """
        Args:
            llm: LangChain ChatModel instance from LLMBaseWrapper
            max_iters: Maximum number of iterations for re-retrieval
        """
        self.llm = llm
        self.max_iters = max_iters
        
        # Generic chat prompt setup
        chat_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["prompt"],
                    template="{prompt}",
                )
            )
        ])
        self._chain = chat_prompt | llm

    def _call_llm(self, prompt: str) -> Tuple[BaseMessage, float, int]:
        """
        Single LLM call with timing and token extraction.
        
        Returns:
            (raw_response, elapsed_time, total_tokens)
        """
        t0 = time.time()
        raw = self._chain.invoke({"prompt": prompt})
        elapsed = time.time() - t0
        tokens = extract_total_tokens(raw)
        return raw, elapsed, tokens

    @abstractmethod
    def scan(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the scan operation.
        
        Returns:
            Dict with keys:
                - result_set: List[Dict] - the retrieved rows
                - time: float - total execution time in seconds
                - tokens: int - total tokens consumed
                - meta: Dict - additional metadata
        """
        pass

    def _get_content(self, raw_response: BaseMessage) -> str:
        """Extract text content from LLM response"""
        return raw_response.content if hasattr(raw_response, "content") else str(raw_response)
