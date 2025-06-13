import os
import math
import subprocess
import pandas as pd


def summarize_data_mgpu(config):
    """
    Splits the input CSV among multiple GPUs, runs hierarchical summarization in parallel,
    and aggregates the results into one CSV. Supports dumping intermediate prompts/summaries per GPU.
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
        end_idx = min((i + 1) * chunk_size, num_rows)
        if start_idx >= num_rows:
            break

        # Slice input for this GPU
        df_slice = df.iloc[start_idx:end_idx].copy()
        partial_input = f"temp_input_{i}_{config['column_name']}.csv"
        partial_output = f"temp_output_{i}_{config['column_name']}.csv"
        partial_inter = f"temp_inter_{i}_{config['column_name']}.csv"
        df_slice.to_csv(partial_input, index=False)
        if config.get("dump_intermediates_path"):
            partial_files.append((partial_input, partial_output, partial_inter))
        else:
            partial_files.append((partial_input, partial_output))

        # Build command for this GPU
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
            "--column_name", config["column_name"]
        ]

        # Optional flags
        if config.get("validate_summary", False):
            cmd.append("--validate_summary")
        if config.get("num_attempts") is not None:
            cmd.extend(["--num_attempts", str(config["num_attempts"])])
        if config.get("word_ratio") is not None:
            cmd.extend(["--word_ratio", str(config["word_ratio"])])

        # Propagate dump_intermediates_path if provided, suffixing by GPU index
        if config.get("dump_intermediates_path"):
            cmd.extend(["--dump_intermediates_path", partial_inter])

        # Set the GPU for this process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        print(f"[summarize_data_mgpu] Launching GPU {i} with command: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.wait()

    # Combine partial outputs
    frames = []
    intermediate_frames = []

    if config.get("dump_intermediates_path"):
        for i, (in_f, out_f, inter_f) in enumerate(partial_files):
            if os.path.exists(out_f):
                df_out = pd.read_csv(out_f)
                frames.append(df_out)
    else:
        for i, (in_f, out_f) in enumerate(partial_files):
            if os.path.exists(out_f):
                df_out = pd.read_csv(out_f)
                frames.append(df_out)

    if config.get("dump_intermediates_path"):
        for i, (in_f, out_f, inter_f) in enumerate(partial_files):
            if os.path.exists(inter_f):
                df_inter = pd.read_csv(inter_f)
                intermediate_frames.append(df_inter)
    if not frames:
        print("[summarize_data_mgpu] No partial results to combine. Exiting.")
        # cleanup
        for in_f, out_f in partial_files:
            if os.path.exists(in_f): os.remove(in_f)
            if os.path.exists(out_f): os.remove(out_f)
        return

    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_csv(config["output"], index=False)
    print(f"[summarize_data_mgpu] All partial summaries combined into {config['output']}." )

    # Merge intermediates if any
    if intermediate_frames:
        all_inters = pd.concat(intermediate_frames, ignore_index=True)
        all_inters.to_csv(config["dump_intermediates_path"], index=False)
        print(f"[summarize_data_mgpu] All intermediate steps combined into {config["dump_intermediates_path"]}.")

    # Clean up temp files
    if config.get("dump_intermediates_path"):
        for in_f, out_f, inter_f in partial_files:
            if os.path.exists(in_f): os.remove(in_f)
            if os.path.exists(out_f): os.remove(out_f)
            if os.path.exists(inter_f): os.remove(inter_f)
    else:
        for in_f, out_f in partial_files:
            if os.path.exists(in_f): os.remove(in_f)
            if os.path.exists(out_f): os.remove(out_f)
    print("[summarize_data_mgpu] Temporary files removed.")
