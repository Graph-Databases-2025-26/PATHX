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


def build_followup_chain(llm_wrapper: LLMBaseWrapper, baseline_type: Literal["SQL", "NL"]):
    """
    Follow-up chain for 'more results'.
    Теперь он получает контекст (dataset, nl/sql, previous_rows) на вход.
    """

    def _build_followup_from_context(ctx: dict) -> dict:
        dataset = ctx["dataset"]
        previous_rows = ctx["previous_rows"]

        nl_question = ctx.get("nl_question")
        sql = ctx.get("sql")

        prompt = build_followup_prompt(
            dataset=dataset,
            previous_rows=previous_rows,
            nl_question=nl_question if baseline_type == "NL" else None,
            sql=sql if baseline_type == "SQL" else None,
        )
        return {"prompt": prompt}

    prompt_builder = RunnableLambda(_build_followup_from_context)

    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["prompt"],
                template="{prompt}",
            )
        )
    ])

    llm = llm_wrapper.get_llm_instance()
    return prompt_builder | chat_prompt | llm

