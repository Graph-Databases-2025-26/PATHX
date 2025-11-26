from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

NL_ROOT = Path("nl_baseline_results")
SQL_ROOT = Path("sql_baseline_results")

NL_ROOT.mkdir(parents=True, exist_ok=True)
SQL_ROOT.mkdir(parents=True, exist_ok=True)


def _get_root_for_baseline(baseline_type: str, provider: str) -> Path:
    """
    Resolve the top-level output directory for a given baseline type.

    - watsonx:
          NL  -> nl_baseline_results/
          SQL -> sql_baseline_results/
      - gemini:
          NL  -> nl_baseline_gemini/
          SQL -> sql_baseline_gemini/
      - azure:
          NL  -> nl_baseline_azure/
          SQL -> sql_baseline_azure/
      - else:
          baseline_results/<PROVIDER>/<BASELINE_TYPE>/
    """
    bt = baseline_type.upper()
    prov = provider.lower()

    # watsonx
    if prov == "watsonx":
        if bt == "NL":
            root = Path("nl_baseline_results")
        elif bt == "SQL":
            root = Path("sql_baseline_results")
        else:
            root = Path("baseline_results") / prov / bt
    # gemini
    elif prov == "gemini":
        if bt == "NL":
            root = Path("nl_baseline_gemini")
        elif bt == "SQL":
            root = Path("sql_baseline_gemini")
        else:
            root = Path("baseline_results") / prov / bt
    # azure
    elif prov == "azure":
        if bt == "NL":
            root = Path("nl_baseline_azure")
        elif bt == "SQL":
            root = Path("sql_baseline_azure")
        else:
            root = Path("baseline_results") / prov / bt
    # fallback for other providers
    else:
        root = Path("baseline_results") / prov / bt

    root.mkdir(parents=True, exist_ok=True)
    return root


def get_baseline_output_dir(
    baseline_type: str,
    dataset: str,
    provider: str,
) -> Path:
    """
    Compute and create (if needed) the directory where a single query's
    JSON result should be stored.

    Final layout:

      nl_baseline_results/
        WATSONX/
          FLIGHT-2/
            query1.json
            query2.json
      sql_baseline_results/
        AZURE/
          IMDB/
            query1.json

    Args:
        baseline_type: "NL", "SQL", etc.
        dataset: dataset name, e.g. "flight-2".
        provider: LLM provider name, e.g. "watsonx", "gemini", "azure".

    Returns:
        Path to the directory for this (baseline_type, provider, dataset).
    """
    root = _get_root_for_baseline(baseline_type, provider)
    dataset_dir = root / dataset.upper()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def save_query_result_json(
    baseline_type: str,
    dataset: str,
    provider: str,
    query_idx: int,
    data: Dict[str, Any],
) -> Path:
    """
    Save a single query result as a JSON file.

    Args:
        baseline_type: "NL" or "SQL".
        dataset: dataset name, e.g. "flight-2".
        provider: provider name, e.g. "watsonx", "gemini", "azure".
        query_idx: 1-based query index (used in filename: query{idx}.json).
        data: the result payload, usually:
              { "result_set": [...], "time": float|None, "tokens": Any }

    Returns:
        The path to the written JSON file.
    """
    out_dir = get_baseline_output_dir(
        baseline_type=baseline_type,
        dataset=dataset,
        provider=provider,
    )

    out_path = out_dir / f"query{query_idx}.json"

    try:
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            f"[{baseline_type} baseline] Saved result for dataset={dataset}, "
            f"provider={provider}, query={query_idx} → {out_path}"
        )
    except Exception as e:
        logger.error(
            f"Failed to write JSON result for dataset={dataset}, "
            f"provider={provider}, query={query_idx}: {e}"
        )
        raise

    return out_path
