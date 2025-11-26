from __future__ import annotations

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Response(BaseModel):
    """
    Unified baseline response model.

    {
      "result_set": [
        { "col1": ..., "col2": ..., ... },
        ...
      ]
    }
    """
    result_set: List[Dict[str, Union[str, int, float, bool, None, Any]]] = Field(
        default_factory=list,
        description="List of result records, each as {column_name: value}.",
    )


PARSER = PydanticOutputParser(pydantic_object=Response)
