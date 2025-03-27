import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple


class CourtDataLoader:
    """Data loader for court decisions and press releases dataset."""

    def __init__(self, data_path: Union[str, Path] = 'data/raw/german_courts.csv'):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the dataset CSV file
        """
        self.data_path = Path(data_path)
        self._data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the court decisions and press releases dataset.

        Returns:
            DataFrame containing the dataset
        """
        try:
            self._data = pd.read_csv(self.data_path)
            print(f"Data loaded: {len(self._data)} entries")
            return self._data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        """Get the loaded dataset or load it if not already loaded."""
        if self._data is None:
            return self.load_data()
        return self._data

    def get_sample(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Get a random sample of the dataset.

        Args:
            n: Number of samples
            random_state: Random seed for reproducibility

        Returns:
            DataFrame containing a sample of the dataset
        """
        return self.data.sample(min(n, len(self.data)), random_state=random_state)

    def save_cleaned_data(self, df: pd.DataFrame, filename: str = "cleaned_combined_methods.csv") -> Path:
        """
        Save cleaned dataset to the data/processed directory.

        Args:
            df: DataFrame to save
            filename: Name of the file

        Returns:
            Path to the saved file
        """
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / filename
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")

        return output_file

    def save_metadata(self, df: pd.DataFrame, filename: str = "cleaning_metadata.csv") -> Path:
        """
        Save metadata about the cleaning process.

        Args:
            df: DataFrame with metadata columns
            filename: Name of the output file

        Returns:
            Path to the saved file
        """
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)

        # Extract metadata columns
        metadata_cols = [
            'id', 'is_announcement_rule', 'tfidf_similarity', 'is_dissimilar_tfidf',
            'is_irrelevant_ml', 'irrelevant_prob', 'cluster', 'is_irrelevant_cluster',
            'irrelevant_votes', 'is_irrelevant_combined'
        ]

        # Filter available columns
        available_cols = [col for col in metadata_cols if col in df.columns]

        metadata_df = df[available_cols]
        output_file = output_dir / filename
        metadata_df.to_csv(output_file, index=False)
        print(f"Cleaning metadata saved to: {output_file}")

        return output_file

    def load_cleaned_data(self):
        """
        Load cleaned dataset from previous run, if available.

        Returns:
            DataFrame with cleaned data if available, None otherwise
        """
        try:
            return pd.read_csv(Path('data/processed') / "cleaned_combined_methods.csv")
        except FileNotFoundError:
            return None
