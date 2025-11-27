from src.config.config import load_config
from src.baselines.llm_factory import get_llm_wrapper
from src.galois.llm_scan import LLMScanOperator


def test_dummy_table_scan():
    config = load_config()
    wrapper = get_llm_wrapper(config)
    llm = wrapper.get_llm_instance()

    op = LLMScanOperator(llm, max_iters=3)

    result = op.table_scan(
        dataset="WORLD",
        table="city",
        attributes=["Name", "CountryCode", "Population"],
        where_clause="Population > 1000000",
    )

    assert "result_set" in result
    assert isinstance(result["result_set"], list)
