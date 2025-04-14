# summarizer_hier/multi_gpu.py

import os
import math
import subprocess
import pandas as pd

def summarize_data_mgpu(args):
    """
    Splits the input CSV among multiple GPUs, runs hierarchical summarization in parallel,
    and aggregates the results into one CSV. Afterwards, it removes the temporary CSV files.
    """

    df = pd.read_csv(args.input)
    num_rows = len(df)

    if num_rows == 0:
        print("[summarize_data_mgpu] The input CSV is empty. Exiting.")
        return

    gpu_count = args.gpu_count
    if gpu_count < 1:
        print(f"[summarize_data_mgpu] Invalid gpu_count={gpu_count}. Using 1 GPU instead.")
        gpu_count = 1

    print(f"[summarize_data_mgpu] Loaded {num_rows} rows from {args.input}.")
    print(f"[summarize_data_mgpu] Splitting data for {gpu_count} GPUs...")

    # Calculate how many rows go to each GPU
    chunk_size = math.ceil(num_rows / gpu_count)
    processes = []
    partial_files = []  # keep track of (input_file, output_file) pairs

    # Create sub-DataFrames and run them in parallel
    for i in range(gpu_count):
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size, num_rows)

        if start_idx >= num_rows:
            break  # No more data

        df_slice = df.iloc[start_idx:end_idx].copy()

        partial_input = f"temp_input_{i}.csv"
        partial_output = f"temp_output_{i}.csv"
        df_slice.to_csv(partial_input, index=False)
        partial_files.append((partial_input, partial_output))

        print(f"[summarize_data_mgpu] GPU {i} -> processing rows [{start_idx}:{end_idx}]")

        # Build the command to call the *existing* single-GPU summarize in cli.py
        if args.validate_summary:
            cmd = [
                "python", "-m", "courtpressger.summarizer_hier.cli", "summarize",
                "--input", partial_input,
                "--output", partial_output,
                "--model", args.model,
                "--chunk_size", str(args.chunk_size),
                "--context_len", str(args.context_len),
                "--summary_len", str(args.summary_len),
                "--validate_summary", str(args.validate_summary),
                "--prompts", args.prompts if args.prompts else "",
                "--num_attempts", str(args.num_attempts),
                "--word_ratio", str(args.word_ratio),
                "--column_name", args.column_name
            ]
        else:
            cmd = [
                "python", "-m", "courtpressger.summarizer_hier.cli", "summarize",
                "--input", partial_input,
                "--output", partial_output,
                "--model", args.model,
                "--chunk_size", str(args.chunk_size),
                "--context_len", str(args.context_len),
                "--summary_len", str(args.summary_len),
                "--prompts", args.prompts if args.prompts else "",
                "--num_attempts", str(args.num_attempts),
                "--word_ratio", str(args.word_ratio),
                "--column_name", args.column_name
            ]


        # Add validate_summary flag only if True
        if args.validate_summary:
            cmd.append("--validate_summary")

        # Restrict this subprocess to GPU i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        # Launch the subprocess
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all subprocesses to finish
    for p in processes:
        p.wait()

    # Combine partial outputs
    frames = []
    for i, (partial_input, partial_output) in enumerate(partial_files):
        if os.path.exists(partial_output):
            df_out = pd.read_csv(partial_output)
            frames.append(df_out)

    if not frames:
        print("[summarize_data_mgpu] No partial results to combine. Exiting.")
        # Clean up any temp files that might exist before returning
        for (in_f, out_f) in partial_files:
            if os.path.exists(in_f):
                os.remove(in_f)
            if os.path.exists(out_f):
                os.remove(out_f)
        return

    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_csv(args.output, index=False)
    print(f"[summarize_data_mgpu] All partial summaries combined into {args.output}.")

    # Clean up temp files
    for (in_f, out_f) in partial_files:
        if os.path.exists(in_f):
            os.remove(in_f)
        if os.path.exists(out_f):
            os.remove(out_f)

    print("[summarize_data_mgpu] Temporary files removed.")
