from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import google.generativeai as genai
import json
import os
import logging
import re
from typing import List, Dict, Any, Callable
import time
from pathlib import Path
from importlib import import_module
import argparse

from .common import discover_datasets, read_queries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


DATA_ROOT = Path("data")
OUT_ROOT = Path("nl_baseline_results")
PROVIDER_CHOICES = ("gemini", "watsonx")

def _load_provider(name: str) -> Callable[[str], Any]:
    
    try:
        if name == "gemini":
            return import_module("src.llm.gemini_client").generate_text
        if name == "watsonx":
            return import_module("src.llm.watsonx_client").generate_text
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM provider '{name}': {e}")
    raise ValueError(f"Unknown provider: {name}")

def _build_prompt(nl_query: str) -> str:
    return f"""
    I will give you a natural language question to answer
Return ONLY the execution result as a valid JSON array.

Natural Language Question:
{nl_query}

"""

def _try_extract_json(text: str) -> List[Dict]:
    
    text = text.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) extract between first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # continue to fallback
            pass

    # 3) fallback object (won't evaluate well, but at least is JSON)
    logger.warning("Failed to parse LLM output as JSON, storing raw text.")
    return [
        {"_error": "failed_to_parse_json",
        "raw_text": text,}
            ]


def _dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


def run_nl_baseline_for_dataset(dataset: str, llm_fn: Callable[[str], Any]) -> None:
    ds_dir = _dataset_dir(dataset)
    if not ds_dir.exists():
        logger.error(f"Dataset directory not found: {ds_dir}")
        return

    out_dir = OUT_ROOT / dataset.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: List[str] = read_queries(ds_dir)
    if not queries:
        logger.warning(f"No queries_*.sql found for dataset {dataset}")
        return

    logger.info(f"[NL baseline] Dataset: {dataset} ({len(queries)} queries)")

    for i, query in enumerate(queries, start=1):
        logger.info(f"[{dataset}] Q{i}/{len(queries)}")
        prompt = _build_prompt(query)

        elapsed = None 

        tokens = None
        try:
            t0 = time.time()
            result = llm_fn(prompt)
            elapsed = time.time() - t0
            if isinstance(result, (tuple, list)):
                raw_answer = result[0]
                tokens = result[1] if len(result) > 1 else None
            else:
                raw_answer = result
        except Exception as e:
            logger.error(f"[{dataset}] Q{i} – LLM call failed: {e}")
            data = {"_error": str(e), "prompt": prompt}
        else:
            data = _try_extract_json(raw_answer)

        # Full JSON wrapping
        if isinstance(data, list):
            data = {"result_set": data}
        elif isinstance(data, dict) and "result_set" not in data:
            data = {"result_set": [data]}

        # Add time/tokens
        if isinstance(data, dict):
            data.setdefault("tokens", tokens)
            # If LLM called successfully — write time, else None
            data["time"] = round(elapsed, 3) if elapsed is not None else None

        out_path = out_dir / f"query{i}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{dataset}] Q{i} → {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Run NL baseline (Baseline 2) with select LLM backend.")
    ap.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset names (as folder names under ./data). "
             "If omitted, all datasets discovered in ./data are used.",
    )
    ap.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=os.getenv("LLM_PROVIDER", "watsonx"),
        help="LLM backend to use (default from LLM_PROVIDER env, falls back to watsonx).",
    )

    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    llm_fn = _load_provider(args.provider)
    logger.info(
        f"Running SQL baseline for datasets: {', '.join(datasets)} "
        f"with provider={args.provider}"
    )

    for ds in datasets:
        run_nl_baseline_for_dataset(ds, llm_fn)


if __name__ == "__main__":
    main()


#watsonx model
# api_key      = "EPt6NeFnd0N3t4TFu9b5ASqTEXZejOX-NHKyr5U5x4rw"
# project_id   = "9275c050-718b-4fc2-99a0-f9c14d1054e7"
# cloud_region = "us-south" 

# creds = Credentials(
#     api_key=api_key,
#     url="https://us-south.ml.cloud.ibm.com"
# )

# model_watsonx = ModelInference(
#     model_id="ibm/granite-3-3-8b-instruct",
#     params={
#         "temperature": 0,
#         "max_new_tokens": 2000,
#         "stop_sequences": ["]\n"],
#         "repetition_penalty": 1.2
#     },
#     credentials=creds,
#     project_id=project_id
# )

# #Gemini model
# API_KEY = "AIzaSyB5bevTQvg-g42NayUHoMM6qBkP-zuv9FM"
# if not API_KEY:
#     raise RuntimeError("GEMINI_API_KEY is not set. Please export it in your environment.")

# genai.configure(api_key=API_KEY)

# MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# model_gemini = genai.GenerativeModel(MODEL_NAME)


# def _build_prompt(nl_query: str) -> str:
#     return f"""
#     You receive a natural language query.
#     Your task is to Return ONLY a single JSON array that directly answers the question.
    
#     STRICT RULES:
#     - Output MUST be valid JSON.
#     - Output MUST be ONLY a JSON array.
#     - DO NOT output explanations.
#     - DO NOT output examples.
#     - DO NOT output Query or Answer lines.
#     - DO NOT output markdown.
#     - DO NOT output code fences like ```json.
#     - DO NOT repeat the query.
#     - DO NOT continue generating after the JSON.
#     - The answer MUST be pure JSON and nothing else.
#     - Don't generate other queries or examples
    
#     Requirements:
#     - The output MUST be a JSON array.
#     - Each element MUST be a JSON object.
#     - All objects MUST contain the exact same set of keys.
#     - Keys MUST be strings.
#     - Values MUST be either strings or numbers.
    
#     Now answer this query by returning ONLY one JSON array that has one or more objects:
#     {nl_query}
#     """

# def _try_extract_json(text: str) -> List[Dict]:
    
#     text = text.strip()

#     # 1) direct parse
#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # 2) extract between first '[' and last ']'
#     start = text.find("[")
#     end = text.rfind("]")
#     if start != -1 and end != -1 and end > start:
#         candidate = text[start : end + 1]
#         try:
#             return json.loads(candidate)
#         except Exception:
#             # continue to fallback
#             pass

#     # 3) fallback object (won't evaluate well, but at least is JSON)
#     return [
#         {"_error": "failed_to_parse_json",
#         "raw_text": text,}
#             ]


# DATA = r"D:/GRAPH DATABASE/pathX/PathX/data"
# SUB = r"D:/GRAPH DATABASE/pathX/PathX/nl-baseline-result"
# # PROMPTS = Path("src/llm/prompts")
# os.makedirs(SUB, exist_ok=True)


# for folder in os.listdir(DATA):
#     capital_folder= folder.upper()
#     folder_path =os.path.join(DATA,folder)

#     if os.path.isdir(folder_path):
#         print(f"processing {folder} folder")

#     for file in os.listdir(folder_path):
#         if file.endswith(".txt"):
#             txt_file= os.path.join(folder_path, file)    

#             with open(txt_file, "r", encoding="utf-8") as f:
#                     lines= f.readlines()
        
#                     for line in lines :
#                         line= line.strip()

#                         if line.startswith("--query"):
#                             query_number= line.removeprefix("--")
#                             current_query= None
#                             continue

#                         if line and current_query is None:
#                             current_query= line


#                         prompt = _build_prompt(current_query)


#                         # Watsonx model
#                         startT_watsonx = time.time()

#                         response_watsonx = model_watsonx.generate(prompt=prompt)
#                         raw_output_watsonx = response_watsonx["results"][0]["generated_text"]
#                         data_from_watsonx= _try_extract_json(raw_output_watsonx)

#                         endT_watsonx = time.time()
#                         elapsedW = endT_watsonx - startT_watsonx
#                         tokensW=len(raw_output_watsonx.split())

#                         json_path_w = os.path.join(SUB, "watsonx", capital_folder, f"{query_number}.json")
#                         os.makedirs(os.path.dirname(json_path_w), exist_ok=True)
                        
#                         payloadW = {
#                             "result_set": data_from_watsonx,
#                             "tokens": tokensW,
#                             "time":elapsedW
#                         }
#                         with open(json_path_w, "w", encoding="utf-8") as f:
#                             json.dump(
#                                 payloadW,
#                                 f,
#                                 indent=4,
#                                 ensure_ascii=False
#                             )



#                         # Gemini model 
#                         startT_gemini = time.time()

#                         response_gemini = model_gemini.generate_content(prompt)
#                         raw_output_gemini= response_gemini.text
#                         data_from_gemini= _try_extract_json(raw_output_gemini)

#                         endT_gemini = time.time()
#                         elapsedG = endT_gemini - startT_gemini
#                         tokensG=len(raw_output_gemini.split())

#                         json_path_m = os.path.join(SUB, "gemini", capital_folder, f"{query_number}.json")
#                         os.makedirs(os.path.dirname(json_path_m), exist_ok=True)
                        
#                         payloadG = {
#                             "result_set": data_from_gemini,
#                             "tokens": tokensG,
#                             "time":elapsedG
#                         }
#                         with open(json_path_m, "w", encoding="utf-8") as f:
#                             json.dump(
#                                 payloadG,
#                                 f,
#                                 indent=4,
#                                 ensure_ascii=False
#                             )
        
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

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--datasets", nargs="*", required=True)
#     ap.add_argument("--model", default="watsonx-llm")
#     ap.add_argument("--temperature", type=float, default=0.0)
#     args = ap.parse_args()

#     client = WatsonXClient(cfg=LLMConfig(model=args.model, temperature=args.temperature))
#     for ds in args.datasets:
#         run_dataset(client, ds)

# if __name__ == "__main__":
#     main()
