#!/usr/bin/env python3
"""
Dataset loader for German Court Press project.

This script loads the german-courts dataset from Hugging Face,
merges all subsets and splits, and saves the result to a CSV file.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of all subsets in the german-courts dataset
COURT_SUBSETS = [
    "bundesarbeitsgericht",
    "bundesfinanzhof",
    "bundesgerichtshof",
    "bundessozialgericht",
    "bundesverfassungsgericht",
    "bundesverwaltungsgericht"
]

# List of all splits
SPLIT_NAMES = ["train", "validation", "test"]


def load_court_datasets():
    """
    Load all subsets from the german-courts dataset on Hugging Face.

    Returns:
        pandas.DataFrame: Merged dataframe containing all court datasets
    """
    all_dfs = []

    logger.info("Loading all subsets from the german-courts dataset...")

    # Iterate through each subset (configuration)
    for court in COURT_SUBSETS:
        try:
            # Load the specific subset
            logger.info(f"Loading {court}...")
            subset_data = load_dataset("rusheeliyer/german-courts", court)

            # Process each split in the dataset
            for split in SPLIT_NAMES:
                if split in subset_data:
                    # Convert to pandas DataFrame
                    split_df = subset_data[split].to_pandas()

                    # Add columns for subset name and split name
                    # Using the capitalized version for better readability in subset_name
                    capitalized_court = court[0].upper() + court[1:]
                    split_df['subset_name'] = capitalized_court
                    split_df['split_name'] = split

                    # Add to our list of dataframes
                    all_dfs.append(split_df)

                    logger.info(
                        f"Loaded {court}/{split} with {len(split_df)} entries")
                else:
                    logger.warning(f"Split {split} not found in {court}")
        except Exception as e:
            logger.error(f"Error loading {court}: {e}")

    # Merge all dataframes into one
    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df


def print_dataset_info(df):
    """
    Print information about the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    logger.info(f"Total entries: {len(df)}")
    logger.info("\nEntries per subset:")
    logger.info(df.groupby('subset_name').size())
    logger.info("\nEntries per split:")
    logger.info(df.groupby('split_name').size())
    logger.info("\nEntries per subset and split:")
    logger.info(df.groupby(['subset_name', 'split_name']).size())
    logger.info("\nSample of the merged dataset:")
    logger.info(df.head())


def save_dataset(df, output_path=None):
    """
    Save the merged dataset to a CSV file.

    Args:
        df (pandas.DataFrame): The dataset to save
        output_path (str, optional): The path where to save the dataset.
                                     Defaults to 'data/german_courts.csv'.

    Returns:
        str: The path where the dataset was saved
    """
    # Create output directory if it doesn't exist
    if output_path is None:
        output_dir = Path('data/raw')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "german_courts.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

    # Save the merged dataset
    df.to_csv(output_path, index=False)
    logger.info(
        f"Saved merged dataset to {output_path} with {len(df)} entries")

    return str(output_path)


def main():
    """Main function to execute the dataset loading and saving process."""
    # Load and merge all court datasets
    merged_df = load_court_datasets()

    # Print information about the merged dataset
    print_dataset_info(merged_df)

    # Save the merged dataset
    save_dataset(merged_df)


if __name__ == "__main__":
    main()
