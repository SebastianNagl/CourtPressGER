#!/usr/bin/env python3
"""
Main entry point for the CourtPressGER package.
"""

import argparse
import sys
import logging
from pathlib import Path

from courtpressger.dataset import load_court_datasets, save_dataset, print_dataset_info

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CourtPressGER - German Court Press dataset processor'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download and process the German Courts dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the processed dataset (default: data/german_courts.csv)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("CourtPressGER - German Court Press dataset processor")

    if args.download:
        logger.info("Downloading and processing the German Courts dataset...")
        df = load_court_datasets()
        print_dataset_info(df)
        output_path = save_dataset(df, args.output)
        logger.info(f"Dataset saved to {output_path}")
    else:
        logger.info(
            "No action specified. Use --download to process the dataset.")
        logger.info("For more information, use --help.")


if __name__ == "__main__":
    main()
