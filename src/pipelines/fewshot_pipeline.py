import argparse, json
from pathlib import Path
from utils.io_utils import write_json, read_sql_statements
from utils.timers import timer
from llm.watsonx_client import WatsonXClient, LLMConfig

DATA = Path("data")
SUB = Path("submissions")
PROMPTS = Path("src/llm/prompts")

def build_fewshot_prompt(sql: str) -> str:
    template = (PROMPTS / "fewshot.txt").read_text(encoding="utf-8")
    return template.replace("{{ sql }}", sql)

def parse_result(text: str) -> list[dict]:
    try:
        arr = json.loads(text)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

def run_dataset(client: WatsonXClient, ds: str):
    ds_dir = DATA / ds
    qfiles = sorted(ds_dir.glob("queries_*.sql"))
    if not qfiles:
        print(f"[WARN] no queries_*.sql in {ds_dir}")
        return

    queries = read_sql_statements(qfiles[0])

    out_dir = SUB / ds.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sql in enumerate(queries, 1):
        prompt = build_fewshot_prompt(sql)

        with timer() as elapsed:
            resp = client.call(prompt)

        rows = parse_result(resp.get("text", "[]"))
        usage = resp.get("usage", {})

        payload = {
            "result_set": rows,
            "time": round(elapsed(), 6),
            "tokens": int(usage.get("tokens_in", 0)) + int(usage.get("tokens_out", 0))
        }

        write_json(out_dir / f"query{i}.json", payload)
        print(f"[FEWSHOT] {ds} query{i}: {len(rows)} rows, {payload['time']} s, tokens={payload['tokens']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", required=True)
    ap.add_argument("--model", default="watsonx-llm")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    client = WatsonXClient(
        cfg=LLMConfig(model=args.model, temperature=args.temperature)
    )

    for ds in args.datasets:
        run_dataset(client, ds)

if __name__ == "__main__":
    main()
