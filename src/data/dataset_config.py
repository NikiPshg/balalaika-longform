#!/usr/bin/env python
"""Single source of truth for the dataset location (configs/dataset.yaml).

    from src.data.dataset_config import load_dataset_config, dataset_root
    cfg = load_dataset_config()        # dict; every path key already absolute
    cfg["root"], cfg["parquet"], cfg["metadata"]["selection_plan"], ...

Overrides: DATASET_CONFIG (another yaml), DATASET_ROOT (only the root, same layout).
Pure stdlib + PyYAML (present in both the cosyvoice env and .venv-eval).
"""
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "dataset.yaml")

_PATH_KEYS_TOP = ("parquet", "audio_dir", "readme", "filter_summary")
_PATH_KEYS_META = ("selection_plan", "catalog_glob", "infojson_dir", "subtitles_dir")


def load_dataset_config(path=None):
    path = path or os.environ.get("DATASET_CONFIG") or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = os.environ.get("DATASET_ROOT") or cfg["root"]
    cfg["root"] = os.path.abspath(root)
    cfg["config_path"] = os.path.abspath(path)
    for k in _PATH_KEYS_TOP:
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(cfg["root"], cfg[k])
    meta = cfg.setdefault("metadata", {})
    for k in _PATH_KEYS_META:
        if k in meta and not os.path.isabs(meta[k]):
            meta[k] = os.path.join(cfg["root"], meta[k])
    if meta.get("layout", "v3") != "v3":
        raise SystemExit(f"[dataset_config] unsupported metadata layout {meta.get('layout')!r}: "
                         "only the v3 layout (selection_plan + catalog_*.json + infojson) is "
                         "supported; the v1 youtube_api_metadata.json layout was dropped on "
                         "2026-08-29")
    return cfg


def dataset_root(path=None):
    return load_dataset_config(path)["root"]


if __name__ == "__main__":
    import json
    print(json.dumps(load_dataset_config(), ensure_ascii=False, indent=2))
