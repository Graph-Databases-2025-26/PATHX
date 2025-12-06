"""
Response parser utilities for extracting JSON from LLM responses
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class ResponseParser:
    """Robust parser for LLM JSON responses"""

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling:
        - Markdown code blocks (```json...```)
        - Plain JSON objects
        - Malformed JSON (best-effort repair)
        - Text before/after JSON
        """
        text = text.strip()
        
        # Remove markdown code blocks
        if "```" in text:
            # Find content between ``` markers
            pattern = r"```(?:json)?\s*(.*?)\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        
        # Try to find ANY JSON object (including nested in text)
        # Look for first { and its matching }
        json_candidates = []
        
        # Method 1: Find all potential JSON objects
        start_positions = [i for i, c in enumerate(text) if c == '{']
        
        for start in start_positions:
            # Count braces to find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        # Found matching brace
                        json_str = text[start:i+1]
                        try:
                            parsed = json.loads(json_str)
                            json_candidates.append(parsed)
                        except json.JSONDecodeError:
                            # Try repair
                            try:
                                repaired = ResponseParser._repair_json(json_str)
                                parsed = json.loads(repaired)
                                json_candidates.append(parsed)
                            except:
                                pass
                        break
        
        if json_candidates:
            # Return first valid JSON found
            return json_candidates[0]
        
        # Method 2: Original method as fallback
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                repaired = ResponseParser._repair_json(json_str)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse JSON: {e}\nOriginal: {json_str[:200]}...")
        
        raise ValueError(f"No valid JSON object found in: {text[:200]}...")

    @staticmethod
    def _repair_json(json_str: str) -> str:
        """
        Best-effort JSON repair for common LLM mistakes:
        - Missing closing brackets
        - Trailing commas
        - Single quotes instead of double quotes
        """
        # Replace single quotes with double quotes (careful with contractions)
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        
        # Remove trailing commas before } or ]
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Try to close unclosed brackets
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)
        
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")
        if open_brackets > close_brackets:
            json_str += "]" * (open_brackets - close_brackets)
        
        return json_str

    @staticmethod
    def extract_result_set(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract result_set from parsed JSON.
        Handles different response formats.
        """
        if "result_set" in response_json:
            result = response_json["result_set"]
            if isinstance(result, list):
                return result
            return [result] if isinstance(result, dict) else []
        
        # Fallback: if response is a list, treat as result_set
        if isinstance(response_json, list):
            return response_json
        
        # Fallback: if response is a single dict, wrap in list
        if isinstance(response_json, dict):
            return [response_json]
        
        return []

    @staticmethod
    def parse_llm_response(
        raw_content: str,
        elapsed: float = 0.0,
        tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Parse LLM response and extract result_set.
        Compatible with src.baselines.core.parsing.parse_llm_response
        
        Returns:
            {
                "result_set": List[Dict],
                "time": float,
                "tokens": int,
            }
        """
        try:
            response_json = ResponseParser.extract_json(raw_content)
            result_set = ResponseParser.extract_result_set(response_json)
            
            return {
                "result_set": result_set,
                "time": elapsed,
                "tokens": tokens or 0,
            }
        except Exception as e:
            print(f"Warning: Failed to parse response: {e}")
            return {
                "result_set": [],
                "time": elapsed,
                "tokens": tokens or 0,
                "error": str(e),
            }

    @staticmethod
    def extract_confidence(response_json: Dict[str, Any]) -> float:
        """Extract confidence value (0.0-1.0) from response"""
        if "confidence" in response_json:
            try:
                return float(response_json["confidence"])
            except (ValueError, TypeError):
                pass
        return 0.5  # Default moderate confidence

    @staticmethod
    def extract_selectivity_map(response_json: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract selectivity evaluations from response.
        Returns mapping: condition -> "higher" or "lower"
        """
        # Response should be flat dict of {condition: selectivity}
        result = {}
        for key, value in response_json.items():
            if key not in ["result_set", "confidence", "error"]:
                if isinstance(value, str) and value.lower() in ["higher", "lower"]:
                    result[key] = value.lower()
        return result

    @staticmethod
    def extract_confidence_map(response_json: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract confidence evaluations from response.
        Returns mapping: condition -> "high" or "low"
        """
        result = {}
        for key, value in response_json.items():
            if key not in ["result_set", "confidence", "error"]:
                if isinstance(value, str) and value.lower() in ["high", "low"]:
                    result[key] = value.lower()
        return result