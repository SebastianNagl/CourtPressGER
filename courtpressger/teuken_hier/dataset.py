"""Dataset preparation for Teuken fine-tuning."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TeukenDatasetPreparer:
    """Prepares datasets for Teuken fine-tuning."""
    
    def __init__(self, tokenizer_name: str = "openGPT-X/Teuken-7B-instruct-research-v0.4", 
                 seed: int = 42):
        """
        Initialize dataset preparer.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            seed: Random seed for reproducibility
        """
        self.tokenizer_name = tokenizer_name
        self.seed = seed
        self.tokenizer = None
    
    def _load_tokenizer(self):
        """Load tokenizer if not already loaded."""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=False,
                trust_remote_code=True
            )
    
    def build_sample(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a training sample with Teuken chat template.
        
        Args:
            row: DataFrame row with prompt and summary
            
        Returns:
            Dictionary with prompt and completion
        """
        self._load_tokenizer()
        
        # Build user message
        user_text = row['prompt']
        
        messages = [
            {"role": "User", "content": user_text}
        ]
        
        # Apply chat template
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            chat_template="DE",
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "prompt": prompt_str,
            "completion": row["summary"]
        }
    
    def prepare_dataset(self, 
                       input_path: str,
                       output_dir: str,
                       train_split: float = 0.9,
                       id_column: str = "id",
                       prompt_column: str = "prompt",
                       summary_column: str = "summary") -> None:
        """
        Prepare dataset from CSV for fine-tuning.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Output directory for processed files
            train_split: Ratio for train/validation split
            id_column: Name of ID column
            prompt_column: Name of prompt column
            summary_column: Name of summary column
        """
        logger.info(f"Loading dataset from {input_path}")
        
        # Load CSV
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Check required columns
        required_columns = [prompt_column, summary_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Rename columns for consistency
        df = df.rename(columns={
            prompt_column: 'prompt',
            summary_column: 'summary'
        })
        
        # If split_name column exists, use it
        if 'split_name' in df.columns:
            train_df = df[df['split_name'] == 'train']
            val_df = df[df['split_name'] == 'validation']
            test_df = df[df['split_name'] == 'test']
            
            logger.info(f"Using existing splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        else:
            # Shuffle and split
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
            split_idx = int(len(df) * train_split)
            train_df = df[:split_idx]
            val_df = df[split_idx:]
            test_df = val_df  # Use validation as test for now
            
            logger.info(f"Created splits - Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process and save each split
        for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            if len(split_df) == 0:
                logger.warning(f"Skipping empty {split_name} split")
                continue
            
            output_file = output_path / f"{split_name}.jsonl"
            
            logger.info(f"Processing {split_name} split ({len(split_df)} samples)")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
                    sample = self.build_sample(row)
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"Saved {split_name} split to {output_file}")
        
        logger.info("Dataset preparation complete")