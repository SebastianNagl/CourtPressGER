"""Data preparation for human evaluation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .config import DataPreparationConfig, ModelSummaryConfig

logger = logging.getLogger(__name__)


class HumanEvalDataPreparer:
    """Prepares data for human evaluation."""
    
    def __init__(self, config: DataPreparationConfig):
        """
        Initialize data preparer.
        
        Args:
            config: Data preparation configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_evaluation_dataset(self) -> Path:
        """
        Prepare base evaluation dataset with sampling.
        
        Returns:
            Path to output JSON file
        """
        logger.info(f"Loading data from {self.config.input_csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.config.input_csv_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Validate columns
        missing_cols = set(self.config.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sample per court
        logger.info(f"Sampling {self.config.sample_size_per_court} cases per court...")
        sampled_df = (
            df.groupby("subset_name")
            .apply(lambda g: g.sample(
                n=min(len(g), self.config.sample_size_per_court),
                random_state=self.config.random_seed
            ))
            .reset_index(drop=True)
        )
        
        logger.info(f"Sampled {len(sampled_df)} total cases")
        
        # Keep only required columns
        sampled_df = sampled_df[self.config.required_columns]
        
        # Convert to JSON
        output_path = self.output_dir / "human_eval_base.json"
        records = sampled_df.to_dict(orient="records")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(records)} records to {output_path}")
        
        # Log distribution
        distribution = sampled_df["subset_name"].value_counts().sort_index()
        logger.info("Court distribution in sample:")
        for court, count in distribution.items():
            logger.info(f"  {court}: {count}")
        
        return output_path
    
    def augment_with_model_summaries(self,
                                    input_path: str,
                                    output_path: str,
                                    model_config: ModelSummaryConfig) -> Path:
        """
        Augment evaluation data with model-generated summaries.
        
        Args:
            input_path: Path to base evaluation JSON
            output_path: Path for augmented output
            model_config: Model summary configuration
            
        Returns:
            Path to output file
        """
        logger.info(f"Loading base data from {input_path}")
        
        # Load base data
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        id_set = {rec["id"] for rec in records}
        logger.info(f"Processing {len(records)} records")
        
        # Build summary lookup
        summary_lookup = {}
        
        for spec in model_config.model_specs:
            csv_path = Path(spec['csv_path'])
            column_name = spec['column_name']
            
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                continue
            
            logger.info(f"Loading summaries from {csv_path.name}")
            
            # Load CSV with only needed columns
            df = pd.read_csv(csv_path, usecols=["id", column_name])
            df = df[df["id"].isin(id_set)]
            
            # Check for missing IDs
            missing = id_set - set(df["id"])
            if missing:
                logger.warning(f"{csv_path.name}: missing {len(missing)} IDs")
            
            # Store in lookup
            summary_lookup[column_name] = df.set_index("id")[column_name].to_dict()
            logger.info(f"  Loaded {len(df)} summaries from column '{column_name}'")
        
        # Augment records
        for rec in records:
            rec_id = rec["id"]
            rec["model_summaries"] = []
            
            for spec in model_config.model_specs:
                column_name = spec['column_name']
                if column_name in summary_lookup and rec_id in summary_lookup[column_name]:
                    rec["model_summaries"].append({
                        "model_name": column_name,
                        "summary": summary_lookup[column_name][rec_id],
                        "category": spec.get('category', 'unknown')
                    })
        
        # Save augmented data
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved augmented data to {output_path}")
        
        # Log summary statistics
        model_counts = {}
        for rec in records:
            for ms in rec.get("model_summaries", []):
                model_name = ms["model_name"]
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        logger.info("Model summary counts:")
        for model, count in sorted(model_counts.items()):
            logger.info(f"  {model}: {count}")
        
        return output_path