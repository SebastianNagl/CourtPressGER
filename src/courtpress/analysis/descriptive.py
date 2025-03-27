"""
This module provides utilities for descriptive analysis of court decisions and press releases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os

# Try to import GPU acceleration
try:
    import cudf
    import cupy as cp
    USE_GPU = True
except ImportError:
    USE_GPU = False


class DescriptiveAnalyzer:
    """
    Class for performing descriptive analysis on court decisions and press releases.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the descriptive analyzer.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and USE_GPU
        self.df = None
        if self.use_gpu:
            print("GPU acceleration enabled for descriptive analysis")
        else:
            print("Using CPU for descriptive analysis")

    def load_data(self, file_path: str = 'data/raw/german_courts.csv') -> pd.DataFrame:
        """
        Load the dataset of court decisions and press releases.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame containing the dataset
        """
        try:
            if self.use_gpu:
                df = cudf.read_csv(file_path)
                print("Dataset loaded with GPU acceleration")
            else:
                df = pd.read_csv(file_path)
                print("Dataset loaded with CPU")

            print(f"Loaded {len(df)} entries")
            self.df = df
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def calculate_basic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic statistics about the dataset.

        Args:
            df: DataFrame containing court decisions and press releases

        Returns:
            DataFrame with basic statistics
        """
        # Add length columns if they don't exist
        if 'summary_length' not in df.columns:
            df['summary_length'] = df['summary'].str.len()

        if 'judgement_length' not in df.columns:
            df['judgement_length'] = df['judgement'].str.len()

        # Calculate basic statistics
        stats = df[['summary_length', 'judgement_length']].describe()

        return stats

    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for duplicates in the dataset.

        Args:
            df: DataFrame to check for duplicates

        Returns:
            Dictionary with duplicate counts
        """
        duplicates = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_ids': df['id'].duplicated().sum()
        }

        return duplicates

    def analyze_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the distribution of cases by year.

        Args:
            df: DataFrame containing court decisions and press releases

        Returns:
            DataFrame with year distribution
        """
        # Extract year from date if not already a column
        if 'year' not in df.columns:
            df['year'] = df['date'].str.extract(r'(\d{4})').astype(int)

        # Count cases by year
        year_counts = df['year'].value_counts().sort_index()

        return year_counts

    def analyze_by_court(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the distribution of cases by court.

        Args:
            df: DataFrame containing court decisions and press releases

        Returns:
            DataFrame with court distribution
        """
        # Count cases by subset_name
        court_counts = df['subset_name'].value_counts()

        return court_counts

    def plot_year_distribution(self, year_counts: pd.Series,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the distribution of cases by year.

        Args:
            year_counts: Series of counts by year
            save_path: Path to save the figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        year_counts.plot(kind='line', marker='o')
        plt.title('Verteilung der Fälle nach Jahr')
        plt.xlabel('Jahr')
        plt.ylabel('Anzahl der Fälle')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return plt.gcf()

    def plot_court_distribution(self, court_counts: pd.Series,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the distribution of cases by court.

        Args:
            court_counts: Series of counts by court
            save_path: Path to save the figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        court_counts.plot(kind='bar')
        plt.title('Verteilung der Fälle nach Gericht')
        plt.xlabel('Gericht')
        plt.ylabel('Anzahl der Fälle')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return plt.gcf()

    def plot_length_distribution(self, df: pd.DataFrame,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot the distribution of text lengths.

        Args:
            df: DataFrame containing summary_length and judgement_length
            save_path: Path to save the figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Create a 1x2 subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot summary length distribution
        sns.histplot(df['summary_length'], kde=True, ax=ax1)
        ax1.set_title('Verteilung der Länge der Pressemitteilungen')
        ax1.set_xlabel('Anzahl der Zeichen')
        ax1.set_ylabel('Häufigkeit')

        # Plot judgement length distribution
        sns.histplot(df['judgement_length'], kde=True, ax=ax2)
        ax2.set_title('Verteilung der Länge der Gerichtsurteile')
        ax2.set_xlabel('Anzahl der Zeichen')
        ax2.set_ylabel('Häufigkeit')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig

    def plot_length_relationship(self, df: pd.DataFrame,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the relationship between summary and judgement lengths.

        Args:
            df: DataFrame containing summary_length and judgement_length
            save_path: Path to save the figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Create scatter plot with regression line
        sns.regplot(x='summary_length', y='judgement_length',
                    data=df, scatter_kws={'alpha': 0.5})
        plt.title(
            'Zusammenhang zwischen Länge der Pressemitteilung und des Gerichtsurteils')
        plt.xlabel('Länge der Pressemitteilung (Zeichen)')
        plt.ylabel('Länge des Gerichtsurteils (Zeichen)')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return plt.gcf()

    def save_analysis_results(self, results: Dict[str, Any],
                              output_dir: str = 'analysis_results') -> Dict[str, str]:
        """
        Save analysis results to CSV files.

        Args:
            results: Dictionary of analysis results
            output_dir: Directory to save results

        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = {}

        for name, data in results.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                file_path = output_path / f"{name}.csv"
                data.to_csv(file_path)
                saved_files[name] = str(file_path)

        return saved_files

    def run_analysis(self, output_dir: str = 'analysis_results',
                     save_figures: bool = True) -> Dict[str, Any]:
        """
        Run a descriptive analysis on the dataset.
        This is an alias for run_complete_analysis for compatibility.

        Args:
            output_dir: Directory to save results
            save_figures: Whether to save figures

        Returns:
            Dictionary with analysis results
        """
        if self.df is None:
            print("No data loaded. Loading default dataset...")
            self.load_data()

        if self.df is None or len(self.df) == 0:
            print("Error: No data available for analysis")
            return {}

        return self.run_complete_analysis(
            df=self.df,
            output_dir=output_dir,
            save_figures=save_figures
        )

    def run_complete_analysis(self, df: Optional[pd.DataFrame] = None,
                              data_path: str = 'data/raw/german_courts.csv',
                              output_dir: str = 'analysis_results',
                              save_figures: bool = True) -> Dict[str, Any]:
        """
        Run a complete descriptive analysis on the dataset.

        Args:
            df: DataFrame containing court decisions and press releases
            data_path: Path to load data if df is None
            output_dir: Directory to save results
            save_figures: Whether to save figures

        Returns:
            Dictionary with analysis results
        """
        # Load data if not provided
        if df is None:
            if self.df is not None:
                df = self.df
            else:
                df = self.load_data(data_path)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run analyses
        results = {}

        # Add summary statistics
        results['total_documents'] = len(df)
        results['unique_courts'] = df['subset_name'].nunique()

        if 'year' in df.columns or 'date' in df.columns:
            year_data = self.analyze_by_year(df)
            results['year_range'] = f"{year_data.index.min()}-{year_data.index.max()}"

        # Add text length statistics
        if 'summary_length' not in df.columns:
            df['summary_length'] = df['summary'].str.len()
        if 'judgement_length' not in df.columns:
            df['judgement_length'] = df['judgement'].str.len()

        results['avg_summary_length'] = df['summary_length'].mean()
        results['avg_judgement_length'] = df['judgement_length'].mean()

        # Basic statistics
        results['basic_stats'] = self.calculate_basic_stats(df)

        # Duplicate check
        results['duplicates'] = self.check_duplicates(df)

        # Year distribution
        results['year_counts'] = self.analyze_by_year(df)

        # Court distribution
        results['court_counts'] = self.analyze_by_court(df)

        # Create figures
        if save_figures:
            # Year distribution plot
            fig_year = self.plot_year_distribution(
                results['year_counts'],
                save_path=f"{output_dir}/year_distribution.png"
            )

            # Court distribution plot
            fig_court = self.plot_court_distribution(
                results['court_counts'],
                save_path=f"{output_dir}/court_distribution.png"
            )

            # Length distribution plot
            fig_length = self.plot_length_distribution(
                df,
                save_path=f"{output_dir}/length_distribution.png"
            )

            # Length relationship plot
            fig_relationship = self.plot_length_relationship(
                df,
                save_path=f"{output_dir}/length_relationship.png"
            )

        # Save CSV results
        saved_files = self.save_analysis_results(results, output_dir)
        results['saved_files'] = saved_files

        return results
