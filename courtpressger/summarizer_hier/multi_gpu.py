# summarizer_hier/multi_gpu.py

import os
import math
import subprocess
import pandas as pd

def summarize_data_mgpu(config):
    """
    Splits the input CSV among multiple GPUs, runs hierarchical summarization in parallel,
    and aggregates the results into one CSV. 
    """
    df = pd.read_csv(config["input"])
    num_rows = len(df)

    if num_rows == 0:
        print("[summarize_data_mgpu] The input CSV is empty. Exiting.")
        return

    gpu_count = config.get("gpu_count", 1)
    if gpu_count < 1:
        print(f"[summarize_data_mgpu] Invalid gpu_count={gpu_count}. Using 1 GPU instead.")
        gpu_count = 1

    print(f"[summarize_data_mgpu] Loaded {num_rows} rows from {config['input']}.")
    print(f"[summarize_data_mgpu] Splitting data for {gpu_count} GPUs...")

    chunk_size = math.ceil(num_rows / gpu_count)
    processes = []
    partial_files = []

    for i in range(gpu_count):
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size, num_rows)
        if start_idx >= num_rows:
            break
        
        df_slice = df.iloc[start_idx:end_idx].copy()
        partial_input = f"temp_input_{i}.csv"
        partial_output = f"temp_output_{i}.csv"
        df_slice.to_csv(partial_input, index=False)
        partial_files.append((partial_input, partial_output))

        cmd = [
            "python", "-m", "courtpressger.summarizer_hier.cli", 
            "--config", config["config"],                       
            "summarize",
            "--input", partial_input,
            "--output", partial_output,
            "--chunk_size", str(config["chunk_size"]),
            "--context_len", str(config["context_len"]),
            "--summary_len", str(config["summary_len"]),
            "--prompts", config["prompts"],
            "--num_attempts", str(config["num_attempts"]),
            "--word_ratio", str(config["word_ratio"]),
            "--column_name", config["column_name"]
        ]

        # If validate_summary is True
        if config.get("validate_summary", False):
            cmd.append("--validate_summary")

        # Restrict to GPU i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    for p in processes:
        p.wait()

    # Combine partial outputs
    frames = []
    for (partial_input, partial_output) in partial_files:
        if os.path.exists(partial_output):
            df_out = pd.read_csv(partial_output)
            frames.append(df_out)

    if not frames:
        print("[summarize_data_mgpu] No partial results to combine. Exiting.")
        for (in_f, out_f) in partial_files:
            if os.path.exists(in_f):
                os.remove(in_f)
            if os.path.exists(out_f):
                os.remove(out_f)
        return

    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_csv(config["output"], index=False)
    print(f"[summarize_data_mgpu] All partial summaries combined into {config['output']}.")

    # Clean up
    for (in_f, out_f) in partial_files:
        if os.path.exists(in_f):
            os.remove(in_f)
        if os.path.exists(out_f):
            os.remove(out_f)
    print("[summarize_data_mgpu] Temporary files removed.")
