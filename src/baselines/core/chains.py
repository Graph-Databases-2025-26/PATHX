from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

from ..llm_provider import LLMBaseWrapper
from .prompt_builder import build_sql_prompt, build_nl_prompt, build_followup_prompt
from .context_builder import get_db_context


def build_lcel_chain(
    llm_wrapper: LLMBaseWrapper,
    baseline_type: Literal["SQL", "NL"],
):
    def _build_prompt_from_context(ctx: dict) -> dict:
        dataset = ctx["dataset"]
        schema_info = ctx["schema_info"]
        strategy = ctx["strategy"]

        if baseline_type == "SQL":
            prompt = build_sql_prompt(strategy, dataset, ctx["sql"], schema_info)
        else:
            prompt = build_nl_prompt(strategy, dataset, ctx["nl_question"], schema_info)

        
        return {"prompt": prompt}

    db_ctx = RunnableLambda(get_db_context)
    prompt_builder = RunnableLambda(_build_prompt_from_context)

    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["prompt"],
                template="{prompt}",
            )
        )
    ])

    llm = llm_wrapper.get_llm_instance()
    return db_ctx | prompt_builder | chat_prompt | llm


def build_followup_chain(llm_wrapper: LLMBaseWrapper):
    """
    Follow-up chain for 'more results'.
    The key here is to prevent ChatPromptTemplate from seeing the JSON curly braces.
    """
    followup_prompt = build_followup_prompt()

    base_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["prompt"],
                template="{prompt}",
            )
        )
    ])

    chat_prompt = base_prompt.partial(prompt=followup_prompt)

    llm = llm_wrapper.get_llm_instance()
    return chat_prompt | llm
