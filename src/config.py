"""Pipeline configuration (config/pipeline.toml) and output paths.

Usage in a script:
    import config
    cfg = config.load()            # or config.load("config/pipeline_smoke.toml")
    cfg["qc"]["vza_max"]
"""

import json
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root; this file is in src/
CONFIG_DIR = ROOT / "config"
DEFAULT = CONFIG_DIR / "pipeline.toml"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
PREPROC_DIR = RESULTS_DIR / "preprocessing"
MODEL_DIR = RESULTS_DIR / "model"
REPORT_DIR = RESULTS_DIR / "report"


def load(path=None):
    """Return the configuration as a dict; path defaults to $OLR_GP_CONFIG or config/pipeline.toml."""
    path = Path(path or os.environ.get("OLR_GP_CONFIG") or DEFAULT)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return cfg


def save(cfg, path):
    """Write the configuration that produced an output next to it, as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cfg, indent=2, default=str))
    os.replace(tmp, path)  # atomic, so concurrent cluster jobs never leave a half-written file
