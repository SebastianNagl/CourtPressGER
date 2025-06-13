#!/usr/bin/env python
"""
augment_with_model_summaries.py

For every ID in the evaluation JSON, attach all model-generated summaries
as an array of {model_name, summary} objects and write a new JSON file.
"""

from pathlib import Path
import json
import pandas as pd

# ----------------------------------------------------------------------
# 0)  Paths
# ----------------------------------------------------------------------
JSON_PATH_IN = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/"
    "cases_prs_synth_prompts_test_subset_human_eval_w_o_model_summaries.json"
)

JSON_PATH_OUT = Path(
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/"
    "cases_prs_synth_prompts_test_subset_human_eval_WITH_model_summaries.json"
)

# ----------------------------------------------------------------------
# 1)  Load the original evaluation records
# ----------------------------------------------------------------------
with JSON_PATH_IN.open(encoding="utf-8") as f:
    records = json.load(f)              # list[dict]

id_set = {rec["id"] for rec in records}
print(f"Loaded {len(records):,} records ({len(id_set):,} unique IDs)")

# ----------------------------------------------------------------------
# 2)  Where to find the model outputs
#     Each tuple:  (csv_path, column_name_with_summary)
# ----------------------------------------------------------------------
model_specs = [
    # FULL GENERATION MODELS
    ("/home/heshmo/workspace/CourtPressGER/data/generation/full/"
     "cases_prs_synth_prompts_test_sample_generated_judgement_summ_llama_3_3_70B.csv",
     "llama_3_3_70B_generated_judgement_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/full/"
     "cases_prs_synth_prompts_test_sample_generated_judgement_summ_mistral_v03.csv",
     "mistral_v03_generated_judgement_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/full/"
     "cases_prs_synth_prompts_test_sample_generated_judgement_summ_openai_gpt_4o.csv",
     "openai_gpt_4o_generated_judgement_summary"),

    # HIERARCHICAL GENERATION MODELS
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_llama_3_3_70B.csv",
     "llama_3_3_70B_gen_hier_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_mistral_v03.csv",
     "mistral_v03_gen_hier_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_openai_gpt_4o.csv",
     "openai_gpt_4o_gen_hier_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_eurollm.csv",
     "eurollm_gen_hier_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_llama_3_8b.csv",
     "llama_3_8b_gen_hier_summary"),
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_teuken.csv",
     "teuken_gen_hier_summary"),
    # FINETUNED TEUKEN
    ("/home/heshmo/workspace/CourtPressGER/data/generation/hier/"
     "cases_prs_synth_prompts_test_sample_hier_gen_summ_teuken_hier_summ-press-summary-v2.csv",
     "teuken_gen_hier_summ_summary-press-summary-v2"),
]

# ----------------------------------------------------------------------
# 3)  Build a lookup:  {model_name -> {id -> summary}}
# ----------------------------------------------------------------------
summary_lookup = {}

for csv_path, summary_col in model_specs:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, usecols=["id", summary_col])
    df = df[df["id"].isin(id_set)]      # keep only the eval IDs

    # sanity check
    missing = id_set.difference(df["id"])
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing {len(missing)} IDs "
            f"(e.g. {sorted(missing)[:5]})"
        )

    summary_lookup[summary_col] = df.set_index("id")[summary_col].to_dict()
    print(f" ↪︎ cached {len(df):,} summaries from {csv_path.name}")

# ----------------------------------------------------------------------
# 4)  Attach summaries to each record
# ----------------------------------------------------------------------
for rec in records:
    rec_id = rec["id"]
    rec["model_summaries"] = [
        {
            "model_name": summary_col,
            "summary":    summary_lookup[summary_col][rec_id],
        }
        for _, summary_col in model_specs
    ]

print("✅ Added model_summaries to every record")

# ----------------------------------------------------------------------
# 5)  Save the enriched JSON
# ----------------------------------------------------------------------
with JSON_PATH_OUT.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"🎉 Wrote {JSON_PATH_OUT}")

# ----------------------------------------------------------------------
# 6)  Inspect: show JSON schema & possible values
# ----------------------------------------------------------------------
print("\n--- JSON structure preview ---")
print("Top-level keys:", list(records[0].keys()))
print("Keys inside model_summaries[0]:", list(records[0]["model_summaries"][0].keys()))

id_values = sorted(id_set)
model_names = [col for _, col in model_specs]

print(f"\nPossible values for 'id' ({len(id_values)}):")
print(id_values)

print(f"\nPossible values for 'model_name' ({len(model_names)}):")
print(model_names)