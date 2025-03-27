#!/usr/bin/env python
"""
Script to run descriptive analysis on court decisions and press releases.

Usage:
    python run_analysis.py [--output-dir OUTPUT_DIR] [--data-file DATA_FILE] [--save-figures]

Options:
    --output-dir OUTPUT_DIR   Directory to save analysis results [default: analysis_results]
    --data-file DATA_FILE     Path to the data file [default: data/raw/german_courts.csv]
    --save-figures            Save generated figures to output directory
"""

import sys
import argparse
import os
from courtpress.analysis import DescriptiveAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run descriptive analysis on court decisions and press releases")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/raw/german_courts.csv",
        help="Path to the data file"
    )

    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save generated figures to output directory"
    )

    return parser.parse_args()


def main():
    """Main function to run descriptive analysis."""
    args = parse_arguments()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        print("Please provide a valid path to the data file.")
        sys.exit(1)

    print(f"Running descriptive analysis on {args.data_file}...")

    # Initialize the analyzer
    analyzer = DescriptiveAnalyzer()

    # Load the data
    analyzer.load_data(args.data_file)

    # Run the analysis
    results = analyzer.run_analysis(
        output_dir=args.output_dir,
        save_figures=args.save_figures
    )

    print("\nAnalysis completed successfully!")
    print(f"Results saved to {args.output_dir}")

    # Print some key statistics
    print("\nKey Statistics:")
    print(
        f"Total number of documents: {results.get('total_documents', 'N/A')}")
    print(f"Number of unique courts: {results.get('unique_courts', 'N/A')}")
    print(f"Year range: {results.get('year_range', 'N/A')}")
    print(
        f"Average judgement length: {results.get('avg_judgement_length', 'N/A'):.2f} characters")
    print(
        f"Average summary length: {results.get('avg_summary_length', 'N/A'):.2f} characters")


if __name__ == "__main__":
    main()
