# Whats my input?
# data/processed/cases_prs_synth_prompts.csv
# model_name
# chunk_size
# context_len
# summary_len
# validate_split

# What are the stages?
# 1) Split the data into chunks
# 2) Hierarichal Summarization of each chunk

# What are the outputs?
# data/processed/cases_prs_synth_prompts_chunked.csv
# data/processed/cases_prs_synth_prompts_hier_summ.csv


import logging
import argparse
from .chunk import chunk_data
from .summ_hier import summarize_data

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the summarizer_hier CLI."""
    parser = argparse.ArgumentParser(description="")
    
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk the data")
    chunk_parser.add_argument("--input", "-i", required=True, help="Input CSV from synthetic_prompts")
    chunk_parser.add_argument("--chunk_size", "-c", type=int, default=2048, help="Chunk size")
    chunk_parser.add_argument("--output", "-o", required=True, help="Output CSV")
    chunk_parser.add_argument("--model", "-m", default="models/eurollm", help="Model to use")
    
    hier_parser = subparsers.add_parser("summarize", help="Hierarichal Summarization")
    hier_parser.add_argument("--input", "-i", required=True, help="Input CSV from chunk")
    hier_parser.add_argument("--output", "-o", required=True, help="Output CSV")
    hier_parser.add_argument("--model", "-m", default="models/eurollm", help="Model to use")
    hier_parser.add_argument("--chunk_size", "-c", type=int, default=2048, help="Chunk size")
    hier_parser.add_argument("--context_len", "-l", type=int, default=4096, help="Context length")
    hier_parser.add_argument("--summary_len", "-s", type=int, default=900, help="Summary length")
    hier_parser.add_argument("--validate_summary", "-v", type=bool, default=False, help="Validate summary")
    hier_parser.add_argument("--prompts", "-p", type=str, help="Prompts to use")
    hier_parser.add_argument("--num_attempts", "-n", type=int, default=3, help="Number of attempts")
    hier_parser.add_argument("--word_ratio", "-w", type=float, default=0.65, help="Word ratio")
    hier_parser.add_argument("--column_name", "-cn", required=True, type=str, help="Column name")


    args = parser.parse_args()

    if args.command == "chunk":
        # python -m courtpressger.summarizer_hier.cli chunk -i /home/heshmo/workspace/CourtPressGER/data/processed/cases_prs_synth_prompts_test_sample.csv -o /home/heshmo/workspace/CourtPressGER/data/processed/cases_prs_synth_prompts_test_sample_chunked.csv -m models/eurollm -c 2048
        chunk_data(args)            
    elif args.command == "summarize":
        # python -m courtpressger.summarizer_hier.cli summarize -i /home/heshmo/workspace/CourtPressGER/data/processed/cases_prs_synth_prompts_test_sample_chunked.csv -o /home/heshmo/workspace/CourtPressGER/data/processed/cases_prs_synth_prompts_test_sample_hier_summ.csv -m models/eurollm -c 2048 -l 4096 -s 900 -p /home/heshmo/workspace/CourtPressGER/prompts
        summarize_data(args)
    
    
    


if __name__ == "__main__":
    main()
