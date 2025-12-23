import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.config import Config_Loader
from src.db import load_queries_from_folder
from src.galois.galois import Galois, save_galois_results
from src.galois.galois_estimator import ConfidenceEstimator
from src.galois.galois_prompts import (
    human_prompt_galois_confidence,
    system_prompt_galois_confidence,
)
from src.utils import DATA_DIR, GROUND_PATH, LOG
from src.utils.tutor.galois_eval import (
    cardinality,
    f1_cell_exact,
    tuple_constraint,
)


# Robustness helpers

def with_retries(fn, *, tries=4, base_sleep=2.0, what="call"):
    last = None
    for t in range(tries):
        try:
            return fn()
        except (
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.ConnectError,
        ) as e:
            last = e
            sleep_s = base_sleep * (2 ** t)
            LOG.warning(
                f"[EXP2] {what} failed ({type(e).__name__}): {e}. "
                f"Retry {t+1}/{tries} in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
    raise last


