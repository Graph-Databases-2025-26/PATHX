from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import google.generativeai as genai
import json
import os
from typing import List, Dict

#watsonx model
api_key      = "EPt6NeFnd0N3t4TFu9b5ASqTEXZejOX-NHKyr5U5x4rw"
project_id   = "9275c050-718b-4fc2-99a0-f9c14d1054e7"
cloud_region = "us-south" 

creds = Credentials(
    api_key=api_key,
    url="https://us-south.ml.cloud.ibm.com"
)

model_watsonx = ModelInference(
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

#Gemini model
API_KEY = "AIzaSyB5bevTQvg-g42NayUHoMM6qBkP-zuv9FM"
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Please export it in your environment.")

genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model_gemini = genai.GenerativeModel(MODEL_NAME)


def _build_prompt(nl_query: str) -> str:
    return f"""
    You receive a natural language query.
    Your task is to Return ONLY a single JSON array that directly answers the question.
    
    STRICT RULES:
    - Output MUST be valid JSON.
    - Output MUST be ONLY a JSON array.
    - DO NOT output explanations.
    - DO NOT output examples.
    - DO NOT output Query or Answer lines.
    - DO NOT output markdown.
    - DO NOT output code fences like ```json.
    - DO NOT repeat the query.
    - DO NOT continue generating after the JSON.
    - The answer MUST be pure JSON and nothing else.
    - Don't generate other queries or examples
    
    Requirements:
    - The output MUST be a JSON array.
    - Each element MUST be a JSON object.
    - All objects MUST contain the exact same set of keys.
    - Keys MUST be strings.
    - Values MUST be either strings or numbers.
    
    Now answer this query by returning ONLY one JSON array that has one or more objects:
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
    return [
        {"_error": "failed_to_parse_json",
        "raw_text": text,}
            ]


DATA = r"D:/GRAPH DATABASE/pathX/PathX/data"
SUB = r"D:/GRAPH DATABASE/pathX/PathX/outputNL"
# PROMPTS = Path("src/llm/prompts")
os.makedirs(SUB, exist_ok=True)


for folder in os.listdir(DATA):
    capital_folder= folder.upper()
    folder_path =os.path.join(DATA,folder)

    if os.path.isdir(folder_path):
        print(f"processing {folder} folder")

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            txt_file= os.path.join(folder_path, file)    

            with open(txt_file, "r", encoding="utf-8") as f:
                    lines= f.readlines()
        
                    for line in lines :
                        line= line.strip()

                        if line.startswith("--query"):
                            query_number= line.removeprefix("--")
                            current_query= None
                            continue

                        if line and current_query is None:
                            current_query= line


                        prompt = _build_prompt(current_query)

                        response_watsonx = model_watsonx.generate(prompt=prompt)
                        raw_output_watsonx = response_watsonx["results"][0]["generated_text"]
                        
                        response_gemini = model_gemini.generate_content(prompt)
                        raw_output_gemini= response_gemini.text
                        print(raw_output_gemini)

                        data= _try_extract_json(raw_output_watsonx)
                        
                        # json_path = os.path.join(SUB, capital_folder, f"{query_number}.json")
                        # os.makedirs(os.path.dirname(json_path), exist_ok=True)
                        # with open(json_path, "w", encoding="utf-8") as f:
                        #     json.dump(
                        #         data,
                        #         f,
                        #         indent=4,
                        #         ensure_ascii=False
                        #     )

        
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
