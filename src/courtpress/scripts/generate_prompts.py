#!/usr/bin/env python3
"""
Court Press Synthetic Prompt Generator

This script generates synthetic prompts that could produce press releases from court decisions
using the Anthropic Claude API.

Usage:
    python generate_prompts.py [--api-key API_KEY] [--model MODEL] [--batch-size BATCH_SIZE] 
                              [--start-idx START_IDX] [--save-interval SAVE_INTERVAL]
                              [--sample-size SAMPLE_SIZE]

Options:
    --api-key           Anthropic API key (defaults to ANTHROPIC_API_KEY env variable)
    --model             Claude model to use (default: claude-3-haiku-20240307)
    --batch-size        Batch size (default: 10)
    --start-idx         Start index for processing (default: 0)
    --save-interval     Save checkpoint interval (default: 10)
    --sample-size       Number of samples to use (default: use all)
"""

import sys
import argparse
import pandas as pd
import os
from pathlib import Path

# Import CourtPress modules
from courtpress.data.loader import CourtDataLoader
from courtpress.models.synthetic_prompts import SyntheticPromptGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Court Press Synthetic Prompt Generator')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (defaults to ANTHROPIC_API_KEY env variable)')
    parser.add_argument('--model', type=str, default='claude-3-haiku-20240307',
                        help='Claude model to use')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index for processing')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint interval')
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Number of samples to use (default: use all)')
    return parser.parse_args()


def main():
    """Run the synthetic prompt generation pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # Check if API key is provided or available as environment variable
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: No Anthropic API key provided. Please provide it with --api-key or set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    # Load data
    print("\n--- Loading Data ---")
    loader = CourtDataLoader()
    df = loader.load_data()

    # Use cleaned data if available
    try:
        cleaned_df = pd.read_csv('data/processed/cleaned_combined_methods.csv')
        print(f"Using cleaned data with {len(cleaned_df)} entries")
        df = cleaned_df
    except FileNotFoundError:
        print("Cleaned data not found, using original data")

    # Use sample if requested
    if args.sample_size > 0:
        if args.sample_size > len(df):
            print(
                f"Sample size {args.sample_size} exceeds dataset size {len(df)}")
            args.sample_size = len(df)

        df = df.sample(args.sample_size, random_state=42)
        print(f"Using sample of {len(df)} entries")

    # Initialize prompt generator
    print("\n--- Initializing Synthetic Prompt Generator ---")
    generator = SyntheticPromptGenerator(api_key=api_key, model=args.model)

    # Generate prompts
    print("\n--- Generating Synthetic Prompts ---")
    results_df = generator.process_batch(
        df=df,
        decision_col='judgement',
        release_col='summary',
        output_col='synthetic_prompt',
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        save_interval=args.save_interval
    )

    # Save final results
    print("\n--- Saving Results ---")
    output_path = generator.save_results(results_df)

    print("\n--- Process Completed Successfully ---")
    print(f"Generated prompts for {len(results_df)} entries")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
