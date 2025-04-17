import os
import glob
import pandas as pd

def aggregate_summaries(args):
    summaries_path = args.summaries_path
    output_path = args.output_path
    origin_csv = args.origin_csv

    # Read the origin CSV
    origin_df = pd.read_csv(origin_csv)

    # Find all CSV files in summaries_path
    csv_files = glob.glob(os.path.join(summaries_path, "*.csv"))

    # Start our merged DataFrame as the origin DataFrame
    merged_df = origin_df.copy()

    # Iterate over each CSV file in the summaries_path
    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)

        # Identify all columns that end with '_summary'
        summary_cols = [c for c in temp_df.columns if c.endswith('_summary')]
        
        # If there's nothing to merge (no summary columns), skip
        if not summary_cols:
            continue

        # We'll merge only the id and the summary columns
        columns_to_merge = ['id'] + summary_cols

        # Merge on 'id', preserving all rows in merged_df
        merged_df = merged_df.merge(temp_df[columns_to_merge], on='id', how='left')

    # Save the final merged DataFrame
    merged_df.to_csv(output_path, index=False)
    print(f"Aggregated summaries saved to: {output_path}")
