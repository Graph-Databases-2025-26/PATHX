def estimate_confidence(llm_client, condition_text: str) -> float:
    """
    Ask the LLM how confident it is about a predicate, return [0..1].
    Stub: returns 0.7 so 'auto' strategies can work during development.
    """
    # prompt = f"Rate confidence 0..1 for correctness of: {condition_text}"
    # resp = llm_client.call(prompt)
    # parse float from resp["text"]
    return 0.7
