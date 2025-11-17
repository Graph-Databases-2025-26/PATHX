from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import json
import os
from typing import List, Dict


api_key      = "EPt6NeFnd0N3t4TFu9b5ASqTEXZejOX-NHKyr5U5x4rw"
project_id   = "9275c050-718b-4fc2-99a0-f9c14d1054e7"
cloud_region = "us-south" 

creds = Credentials(
    api_key=api_key,
    url="https://us-south.ml.cloud.ibm.com"
)

model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    params={
        "temperature": 0,
        "max_new_tokens": 2000,
        "stop_sequences": ["]\n"],
        "repetition_penalty": 1.2
    },
    credentials=creds,
    project_id=project_id
)

DATA = Path("data")
SUB = Path("submissions")
PROMPTS = Path("src/llm/prompts")



# def run_dataset(client: WatsonXClient, ds: str):
#     ds_dir = DATA / ds
#     qfiles = sorted(ds_dir.glob("queries_*.sql"))
#     if not qfiles:
#         print(f"[WARN] no queries_*.sql in {ds_dir}")
#         return
#     queries = read_sql_statements(qfiles[0])

#     out_dir = SUB / ds.upper()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for i, sql in enumerate(queries, 1):
#         prompt = sql_to_nl_prompt(sql)
#         with timer() as elapsed:
#             resp = client.call(prompt)
#         rows = parse_result(resp.get("text", "[]"))
#         usage = resp.get("usage", {})
#         payload = {
#             "result_set": rows,
#             "time": round(elapsed(), 6),
#             "tokens": int(usage.get("tokens_in", 0)) + int(usage.get("tokens_out", 0))
#         }
#         write_json(out_dir / f"query{i}.json", payload)
#         print(f"[NL] {ds} query{i}: {len(rows)} rows, {payload['time']} s, tokens={payload['tokens']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", required=True)
    ap.add_argument("--model", default="watsonx-llm")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    client = WatsonXClient(cfg=LLMConfig(model=args.model, temperature=args.temperature))
    for ds in args.datasets:
        run_dataset(client, ds)

if __name__ == "__main__":
    main()
