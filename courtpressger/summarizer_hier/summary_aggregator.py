"""Module for aggregating summaries from multiple files."""

import logging
import glob
from pathlib import Path
from typing import Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_summaries(input_dir: str, 
                       output_path: str,
                       original_path: Optional[str] = None,
                       pattern: str = "*.csv") -> None:
    """
    Aggregate summaries from multiple CSV files.
    
    Args:
        input_dir: Directory containing summary files
        output_path: Path for aggregated output
        original_path: Optional path to original dataset for merging
        pattern: File pattern to match (default: *.csv)
    """
    logger.info(f"Aggregating summaries from {input_dir}")
    
    # Find all CSV files
    input_path = Path(input_dir)
    csv_files = list(input_path.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} files to aggregate")
    
    # Start with original dataset if provided
    if original_path:
        logger.info(f"Loading original dataset from {original_path}")
        merged_df = pd.read_csv(original_path)
    else:
        # Start with empty dataframe
        merged_df = pd.DataFrame()
    
    # Process each file
    summary_columns = []
    
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Find summary columns (ending with _summary)
            summary_cols = [col for col in df.columns if col.endswith('_summary')]
            
            if not summary_cols:
                logger.warning(f"No summary columns found in {csv_file.name}")
                continue
            
            summary_columns.extend(summary_cols)
            
            # Merge columns
            if 'id' in df.columns:
                # Merge on ID
                cols_to_merge = ['id'] + summary_cols
                
                if merged_df.empty:
                    merged_df = df[cols_to_merge]
                else:
                    merged_df = merged_df.merge(
                        df[cols_to_merge], 
                        on='id', 
                        how='left',
                        suffixes=('', '_dup')
                    )
                    
                    # Handle duplicate columns
                    for col in merged_df.columns:
                        if col.endswith('_dup'):
                            base_col = col[:-4]
                            if base_col in merged_df.columns:
                                # Fill missing values from duplicate
                                merged_df[base_col] = merged_df[base_col].fillna(merged_df[col])
                            merged_df.drop(col, axis=1, inplace=True)
            else:
                logger.warning(f"No 'id' column in {csv_file.name}, skipping merge")
                
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
    
    # Save aggregated results
    if not merged_df.empty:
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Aggregated {len(summary_columns)} summary columns")
        logger.info(f"Results saved to {output_path}")
    else:
        logger.warning("No data to aggregate")


def run_aggregation(input_dir: str, 
                   output_path: str,
                   original_path: Optional[str] = None) -> None:
    """
    Run the aggregation process.
    
    Args:
        input_dir: Directory containing summary files  
        output_path: Path for aggregated output
        original_path: Optional path to original dataset
    """
    aggregate_summaries(input_dir, output_path, original_path)