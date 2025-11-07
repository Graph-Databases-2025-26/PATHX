#!/usr/bin/env python3
from __future__ import annotations
import argparse
from src.llm.watsonx_client import generate_text

def main():
    ap = argparse.ArgumentParser(description="Watsonx.ai smoke test")
    ap.add_argument("--prompt", default="Say hello from watsonx.ai in one short line.")
    ap.add_argument("--model-id", help="Override model id")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args()

    txt = generate_text(
        args.prompt,
        model_id=args.model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n=== watsonx.ai response ===")
    print(txt)

if __name__ == "__main__":
    main()
