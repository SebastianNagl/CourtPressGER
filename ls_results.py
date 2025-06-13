#!/usr/bin/env python3
"""
Merge Label-Studio annotations for court-press evaluation.

* Keeps **only the newest annotation per real case**
  (highest `updated_at` timestamp).
* Validates required fields.
* Flattens to per-press rows and aggregates model metrics.

Author: you 😎
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timezone           # NEW

# ── 1. file locations ────────────────────────────────────────────────
LS_RESULTS_PATH = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/",
    "dummy_ls.json",
)
PRESS_MAP_PATH = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/",
    "cases_prs_synth_prompts_test_subset_human_eval_WITH_model_summaries_PRESS_MODEL_MAP.json",
)
OUT_PATH = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/",
    "output_press_labels_studio.json",
)
METRICS_PATH = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/",
    "per_model_metrics.json",
)

# ── 2. constants ─────────────────────────────────────────────────────
RANK_EXPECTED = 11
QUAL_PREFIXES = ("hallucination", "incoherent", "publishable")

# ── 3.a  timestamp & de-dup helpers ──────────────────────────────────
def _parse_ts(ts: str) -> datetime:
    """
    Convert the ISO timestamp coming from Label-Studio
    (may end with “Z” → UTC) into a timezone-aware datetime.
    """
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def keep_latest_only(tasks: List[dict]) -> List[dict]:
    """
    For every real case (keyed by task['case_id'] if present,
    otherwise by the LS internal task['id']) **keep only the task
    whose `updated_at` is the most recent**.
    """
    newest: dict[str, tuple[datetime, dict]] = {}

    for t in tasks:
        cid = t.get("case_id") or str(t["id"])
        ts = _parse_ts(t["updated_at"])

        if cid not in newest or ts > newest[cid][0]:
            newest[cid] = (ts, t)

    return [pair[1] for pair in newest.values()]


# ── 3.b  validation helpers ──────────────────────────────────────────
def _missing_keys(task: dict) -> List[str]:
    """
    Return a list of attribute-paths that are present but empty.
    Keys that are absent are OK (interpreted as False later).
    """
    miss: List[str] = []

    # rank-block
    rank_block = task.get("rank")
    if not isinstance(rank_block, list) or not rank_block:
        miss.append("rank")
    else:
        rank_list = rank_block[0].get("ranker", {}).get("rank")
        if not isinstance(rank_list, list) or len(rank_list) != RANK_EXPECTED:
            miss.append("rank[0].ranker.rank")

    # per-summary quality fields (1 … 11)
    for i in range(1, RANK_EXPECTED + 1):
        for p in QUAL_PREFIXES:
            key = f"{p}{i}"
            if key in task and task[key] in (None, ""):
                miss.append(key)

    return miss


def validate_tasks(tasks: List[dict]) -> Dict[str, List[str]]:     # NEW
    """return {case_id → [missing keys]} if any problems are found"""
    problems: Dict[str, List[str]] = {}
    for idx, t in enumerate(tasks):
        missing = _missing_keys(t)
        if missing:
            cid = t.get("case_id") or t.get("id") or f"#{idx}"
            problems[cid] = missing
    return problems


# ── 4. merge helpers ────────────────────────────────────────────────
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def bool_from_value(prefix: str, value: str | None) -> bool:
    """
    Convert Label-Studio annotation strings (or missing value) to bools.
    Missing keys arrive here as `None`, interpreted as False.
    """
    txt = (value or "").lower()

    if prefix == "hallucination":
        return "hallucination" in txt or "outside of case" in txt
    if prefix == "incoherent":
        return "incoherent" in txt
    if prefix == "publishable":
        return "publishable" in txt

    return False  # fallback


def build_flat_rows(ls_tasks, press_map):
    # quick lookup (case_id, press_id) → model_name
    model_for = {
        (row["case_id"], row["press_id"]): row["model_name"] for row in press_map
    }

    rows = []
    for task in ls_tasks:
        case_id = task["case_id"]
        rank_list = task["rank"][0].get("ranker", {}).get("rank", [])
        rank_pos = {pid: idx + 1 for idx, pid in enumerate(rank_list)}

        for pid, pos in rank_pos.items():
            pr_num = pid[2:]  # "pr7" → "7"
            row = {
                "case_id": case_id,
                "press_id": pid,
                "model_name": model_for.get((case_id, pid), "UNKNOWN"),
                "rank": pos,
            }
            for pref in QUAL_PREFIXES:
                row[pref] = bool_from_value(pref, task.get(f"{pref}{pr_num}"))
            rows.append(row)
    return rows


def to_hierarchical(rows: List[dict]) -> List[dict]:
    """convert flat rows → per-case hierarchy"""
    bucket: dict[str, list[dict]] = {}
    for r in rows:
        case_id = r["case_id"]
        press_info = {k: v for k, v in r.items() if k != "case_id"}
        bucket.setdefault(case_id, []).append(press_info)

    for lst in bucket.values():
        lst.sort(key=lambda x: x["rank"])

    return [{"case_id": cid, "press_items": items} for cid, items in bucket.items()]


def compute_model_metrics(rows: List[dict]) -> dict:
    """Aggregate statistics per model_name."""
    acc = defaultdict(
        lambda: {
            "count": 0,
            "rank_sum": 0,
            "hallucination_true": 0,
            "incoherent_true": 0,
            "publishable_true": 0,
            "ranks": [], 
        }
    )

    for row in rows:
        m = row["model_name"]
        a = acc[m]
        a["count"] += 1
        a["rank_sum"] += row["rank"]
        a["ranks"].append(row["rank"])
        for p in QUAL_PREFIXES:
            if row[p]:
                a[f"{p}_true"] += 1

    metrics = {}
    for model, a in acc.items():
        c = a["count"]
        metrics[model] = {
            "count": c,
            "avg_rank": round(a["rank_sum"] / c, 3),
            "hallucination_rate": round(a["hallucination_true"] / c, 3),
            "incoherent_rate": round(a["incoherent_true"] / c, 3),
            "publishable_rate": round(a["publishable_true"] / c, 3),
            "ranks": a["ranks"], 
        }
    return metrics


# ── 5. run ───────────────────────────────────────────────────────────
# 5.1 load and **deduplicate**
raw_tasks = load_json(LS_RESULTS_PATH)
ls_tasks = keep_latest_only(raw_tasks)             # NEW

# 5.2 validate the filtered tasks
problems = validate_tasks(ls_tasks)                # CHG
if problems:
    print("❌ Validation failed – missing keys:")
    for cid, keys in problems.items():
        print(f"  • {cid}: {', '.join(keys)}")
    sys.exit(1)

print("✅ Label-Studio file looks good, merging …")

# 5.3 rest of the pipeline
press_map = load_json(PRESS_MAP_PATH)

flat_rows = build_flat_rows(ls_tasks, press_map)
hier_rows = to_hierarchical(flat_rows)
model_metrics = compute_model_metrics(flat_rows)

# 5.4 write outputs
OUT_PATH.write_text(
    json.dumps(hier_rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
METRICS_PATH.write_text(
    json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"✅ wrote {len(hier_rows):,} cases → {OUT_PATH}")
print(f"✅ wrote {len(model_metrics):,} models → {METRICS_PATH}")
