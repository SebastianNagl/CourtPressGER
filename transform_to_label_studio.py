#!/usr/bin/env python
from pathlib import Path
import json
import csv
import random
# ----------------------------------------------------------------------
# 1)  Where are the files?
# ----------------------------------------------------------------------
JSON_IN  = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/"
    "cases_prs_synth_prompts_test_subset_human_eval_WITH_model_summaries.json"
)
JSON_OUT_TASKS = JSON_IN.with_name(JSON_IN.stem + "_LABEL_STUDIO.json")
JSON_OUT_MAP   = JSON_IN.with_name(JSON_IN.stem + "_PRESS_MODEL_MAP.json")

# reproducible randomness
RNG = random.Random(42)

# ----------------------------------------------------------------------
# 2)  Load the input file
# ----------------------------------------------------------------------
with JSON_IN.open(encoding="utf-8") as f:
    records = json.load(f)    # list[dict]

print(f"Loaded {len(records):,} evaluation items from {JSON_IN.name}")


# ----------------------------------------------------------------------
# 4)  Convert to Label-Studio schema
# ----------------------------------------------------------------------
ls_tasks, mapping_rows = [], []

for rec in records:
    case_id   = rec["id"]
    prompt    = rec["synthetic_prompt"]
    ruling    = rec["judgement"]
    court     = rec["subset_name"]

    # build a list of (model_name, summary_text) tuples
    texts = [(ms["model_name"], ms["summary"]) for ms in rec["model_summaries"]]
    texts.append(("reference_summary", rec.get("summary", "")))  # add human/ref

    # ── 1-10: SHUFFLED model summaries ──────────────────────────────
    RNG.shuffle(texts)                     # in-place, reproducible


    press_items, press_flat = [], {}

    for idx, (model_name, body) in enumerate(texts, start=1):
        pid, title = f"pr{idx}", f"PR-{idx}"
        press_items.append({"id": pid, "title": title, "body": body})
        press_flat[f"press{idx}"] = f"{title}\n\n{body}"
        mapping_rows.append({"case_id": case_id,
                                "press_id": pid,
                                "model_name": model_name})

    task_obj = {
        "case_id": case_id,
        "task":      f"Rank & evaluate {len(press_items)} AI-generated press releases.",
        "prompt":    prompt,
        "court_ruling": ruling,
        "issuing_court": court,
        "press_items":   press_items,
        **press_flat,               # press1 … pressN
    }

    ls_tasks.append(task_obj)

# ----------------------------------------------------------------------
# 5)  Save
# ----------------------------------------------------------------------
JSON_OUT_TASKS.write_text(
    json.dumps(ls_tasks, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("✅ tasks  →", JSON_OUT_TASKS)

JSON_OUT_MAP.write_text(
    json.dumps(mapping_rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("✅ mapping →", JSON_OUT_MAP)