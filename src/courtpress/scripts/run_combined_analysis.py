#!/usr/bin/env python3
"""
Combined Analysis Script for Court Press Data

This script combines descriptive analysis and synthetic prompt generation
for court decisions and press releases.

Usage:
    python run_combined_analysis.py [--descriptive] [--synthetic] [--api-key API_KEY]
                                  [--output-dir OUTPUT_DIR] [--sample-size SAMPLE_SIZE]
                                  [--save-figures]

Options:
    --descriptive       Run descriptive analysis
    --synthetic         Run synthetic prompt generation
    --api-key           Anthropic API key for synthetic prompts (defaults to ANTHROPIC_API_KEY env variable)
    --output-dir        Directory to save analysis results (default: analysis_results)
    --sample-size       Number of samples to use (default: use all)
    --save-figures      Save generated figures from descriptive analysis
"""

import sys
import argparse
import pandas as pd
import os
from pathlib import Path

# Import CourtPress modules
from courtpress.data.loader import CourtDataLoader
from courtpress.models.synthetic_prompts import SyntheticPromptGenerator
from courtpress.analysis.descriptive import DescriptiveAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Combined Analysis for Court Press Data')

    # Analysis types
    parser.add_argument('--descriptive', action='store_true',
                        help='Run descriptive analysis')
    parser.add_argument('--synthetic', action='store_true',
                        help='Run synthetic prompt generation')

    # Common parameters
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Number of samples to use (default: use all)')

    # Descriptive analysis parameters
    parser.add_argument('--save-figures', action='store_true',
                        help='Save generated figures from descriptive analysis')

    # Synthetic prompts parameters
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key for synthetic prompts')
    parser.add_argument('--model', type=str, default='claude-3-haiku-20240307',
                        help='Claude model to use for synthetic prompts')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for synthetic prompt generation')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint interval for synthetic prompts')

    return parser.parse_args()


def prepare_data(sample_size=0):
    """Load and prepare the dataset."""
    # Load data
    print("\n--- Loading Data ---")
    loader = CourtDataLoader()
    df = loader.load_data()

    # Use cleaned data if available
    try:
        cleaned_df_path = Path('data/processed/cleaned_combined_methods.csv')
        if cleaned_df_path.exists():
            cleaned_df = pd.read_csv(cleaned_df_path)
            print(f"Using cleaned data with {len(cleaned_df)} entries")
            df = cleaned_df
        else:
            print("Cleaned data not found, using original data")
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        print("Using original data")

    # Use sample if requested
    if sample_size > 0:
        if sample_size > len(df):
            print(f"Sample size {sample_size} exceeds dataset size {len(df)}")
            sample_size = len(df)

        df = df.sample(sample_size, random_state=42)
        print(f"Using sample of {len(df)} entries")

    return df


def run_descriptive_analysis(df, output_dir, save_figures):
    """Run descriptive analysis on the data."""
    print("\n--- Running Descriptive Analysis ---")

    # Initialize analyzer
    analyzer = DescriptiveAnalyzer()

    # Load the data
    analyzer.df = df.copy()

    # Run analysis
    results = analyzer.run_analysis(
        output_dir=output_dir,
        save_figures=save_figures
    )

    print(f"Descriptive analysis completed. Results saved to {output_dir}")
    return results


def run_synthetic_prompt_generation(df, api_key, model, batch_size, save_interval, output_dir):
    """Run synthetic prompt generation."""
    print("\n--- Running Synthetic Prompt Generation ---")

    # Check if API key is provided or available as environment variable
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: No Anthropic API key provided. Please provide it with --api-key or set the ANTHROPIC_API_KEY environment variable.")
        return None

    # Initialize prompt generator
    generator = SyntheticPromptGenerator(api_key=api_key, model=model)

    # Generate prompts
    results_df = generator.process_batch(
        df=df,
        decision_col='judgement',
        release_col='summary',
        output_col='synthetic_prompt',
        batch_size=batch_size,
        start_idx=0,
        save_interval=save_interval,
        checkpoint_dir=os.path.join(output_dir, 'synthetic_checkpoints')
    )

    # Save final results
    output_path = generator.save_results(
        results_df,
        output_file="synthetic_prompts_final.csv",
        output_dir=os.path.join(output_dir, 'synthetic_prompts')
    )

    print(
        f"Synthetic prompt generation completed. Results saved to {output_path}")
    return results_df


def main():
    """Main function to run combined analysis."""
    args = parse_arguments()

    # Check if at least one analysis type is selected
    if not (args.descriptive or args.synthetic):
        print(
            "Error: Please select at least one analysis type (--descriptive or --synthetic)")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepare data
    df = prepare_data(args.sample_size)

    # Run selected analyses
    results = {}

    if args.descriptive:
        desc_results = run_descriptive_analysis(
            df=df,
            output_dir=os.path.join(args.output_dir, 'descriptive'),
            save_figures=args.save_figures
        )
        results['descriptive'] = desc_results

    if args.synthetic:
        synth_results = run_synthetic_prompt_generation(
            df=df,
            api_key=args.api_key,
            model=args.model,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            output_dir=args.output_dir
        )
        results['synthetic'] = synth_results is not None

    print("\n--- Combined Analysis Completed ---")
    print(f"All results saved to: {args.output_dir}")

    # Print summary
    if args.descriptive and 'descriptive' in results:
        print("\nDescriptive Analysis Summary:")
        print(
            f"Total documents analyzed: {results['descriptive'].get('total_documents', 'N/A')}")
        print(
            f"Unique courts: {results['descriptive'].get('unique_courts', 'N/A')}")
        print(f"Year range: {results['descriptive'].get('year_range', 'N/A')}")

    if args.synthetic and 'synthetic' in results:
        print("\nSynthetic Prompt Generation Summary:")
        print(f"Successfully generated prompts: {results['synthetic']}")


if __name__ == "__main__":
    main()
