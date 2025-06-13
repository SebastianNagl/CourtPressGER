import pandas as pd
import json

# 1️⃣ Load the full file
csv_path = "/home/heshmo/workspace/CourtPressGER/data/processed/cases_prs_synth_prompts_test_subset.csv"
df = pd.read_csv(csv_path)

# 2️⃣ Deterministic ≤10-row sample per subset_name
SEED = 42
sampled_df = (
    df.groupby("subset_name")
      .apply(lambda g: (
          g.sample(n=min(len(g), 10), random_state=SEED)  # sample inside group
           .assign(subset_name=g.name)                    # restore the key as a column
      ))
      .reset_index(drop=True)
)

# 3️⃣ Keep only the requested columns
cols = ["id", "subset_name", "synthetic_prompt", "judgement", "summary"]
sampled_df = sampled_df[cols]

# 4️⃣ Write JSON array of objects
out_json = (
    "/home/heshmo/workspace/CourtPressGER/data/human_eval/"
    "cases_prs_synth_prompts_test_subset_human_eval_w_o_model_summaries.json"
)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(sampled_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print(f"✔️  Saved {len(sampled_df)} sampled rows as JSON to {out_json}")

# 5️⃣ Show the subset distribution in the sample
print("\nSubsets represented in the sample:")
for subset, count in sampled_df["subset_name"].value_counts().sort_index().items():
    print(f"  • {subset}: {count}")
