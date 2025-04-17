# summarizer_hier/cli.py

import logging
import argparse
import json
import os
from dotenv import load_dotenv
from .chunk import chunk_data
from .summ_hier import summarize_data
from .multi_gpu import summarize_data_mgpu
from .aggregate_summaries import aggregate_summaries
# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_args_with_config(cli_args: dict, config_section: dict) -> dict:
    """
    Merge CLI args with just the subcommand’s relevant config.
    CLI overrides where applicable.
    """
    # We'll copy so as not to mutate the original.
    final_cfg = config_section.copy()

    for k, v in cli_args.items():
        # skip keys that are None or that are not relevant
        if v is None or k in ["command"]:
            continue
        final_cfg[k] = v

    return final_cfg

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Summarizer CLI with per-command config.")
    
    subparsers = parser.add_subparsers(dest="command")

    # A shared --config just once for all subcommands.
    parser.add_argument("--config", type=str, default=None, help="Path to master config JSON")

    # ---- chunk ----
    chunk_parser = subparsers.add_parser("chunk", help="Chunk the data")
    chunk_parser.add_argument("--input", "-i", required=False)
    chunk_parser.add_argument("--output", "-o", required=False)
    chunk_parser.add_argument("--model", "-m", default=None)
    chunk_parser.add_argument("--tokenizer_name", "-t", default=None)
    chunk_parser.add_argument("--chunk_size", "-c", type=int, default=None)

    # ---- summarize ----
    hier_parser = subparsers.add_parser("summarize", help="Hierarchical Summarization")
    hier_parser.add_argument("--input", "-i", required=False)
    hier_parser.add_argument("--output", "-o", required=False)
    hier_parser.add_argument("--model", "-m", default=None)
    hier_parser.add_argument("--tokenizer_name", "-t", default=None)
    hier_parser.add_argument("--chunk_size", "-c", type=int, default=None)
    hier_parser.add_argument("--context_len", "-l", type=int, default=None)
    hier_parser.add_argument("--summary_len", "-s", type=int, default=None)
    hier_parser.add_argument("--validate_summary", "-v", type=bool, default=None)
    hier_parser.add_argument("--prompts", "-p", type=str, default=None)
    hier_parser.add_argument("--num_attempts", "-n", type=int, default=None)
    hier_parser.add_argument("--word_ratio", "-w", type=float, default=None)
    hier_parser.add_argument("--column_name", "-cn", type=str, default=None)

    # ---- summarize_mgpu ----
    mgpu_parser = subparsers.add_parser("summarize_mgpu", help="Multi-GPU Summarization")
    mgpu_parser.add_argument("--input", "-i", required=False)
    mgpu_parser.add_argument("--output", "-o", required=False)
    mgpu_parser.add_argument("--model", "-m", default=None)
    mgpu_parser.add_argument("--chunk_size", "-c", type=int, default=None)
    mgpu_parser.add_argument("--context_len", "-l", type=int, default=None)
    mgpu_parser.add_argument("--summary_len", "-s", type=int, default=None)
    mgpu_parser.add_argument("--validate_summary", "-v", type=bool, default=None)
    mgpu_parser.add_argument("--prompts", "-p", type=str, default=None)
    mgpu_parser.add_argument("--num_attempts", "-n", type=int, default=None)
    mgpu_parser.add_argument("--word_ratio", "-w", type=float, default=None)
    mgpu_parser.add_argument("--column_name", "-cn", type=str, default=None)
    mgpu_parser.add_argument("--tokenizer_name", "-t", default=None)
    mgpu_parser.add_argument("--gpu_count", type=int, default=None)

    # ---- aggregate_summaries ----
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate Summaries")
    agg_parser.add_argument("--origin_csv", "-oc", type=str, default=None)
    agg_parser.add_argument("--summaries_path", "-s", type=str, default=None)
    agg_parser.add_argument("--output_path", "-o", type=str, default=None)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # load the master config if provided
    master_config = {}
    if args.config:
        master_config = load_config_file(args.config)

    # pick out the sub-command config if it exists
    command_config = master_config.get(args.command, {})
    
    # merge CLI overrides into the command config
    final_config = merge_args_with_config(vars(args), command_config)
    # Dispatch
    if args.command == "chunk":
        chunk_data(final_config)
    elif args.command == "summarize":
        summarize_data(final_config)
    elif args.command == "summarize_mgpu":
        summarize_data_mgpu(final_config)
    elif args.command == "aggregate":
        aggregate_summaries(args)

if __name__ == "__main__":
    main()