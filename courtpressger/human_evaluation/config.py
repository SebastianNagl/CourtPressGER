"""Configuration for human evaluation module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


@dataclass
class DataPreparationConfig:
    """Configuration for data preparation."""
    input_csv_path: str = "data/processed/cases_prs_synth_prompts_test_subset.csv"
    output_dir: str = "data/human_eval"
    sample_size_per_court: int = 10
    random_seed: int = 42
    required_columns: List[str] = field(default_factory=lambda: [
        "id", "subset_name", "synthetic_prompt", "judgement", "summary"
    ])


@dataclass
class ModelSummaryConfig:
    """Configuration for model summary augmentation."""
    model_specs: List[Dict[str, str]] = field(default_factory=list)
    output_dir: str = "data/human_eval"
    
    def add_model_spec(self, csv_path: str, column_name: str, category: str = "unknown"):
        """Add a model specification."""
        self.model_specs.append({
            "csv_path": csv_path,
            "column_name": column_name,
            "category": category
        })


@dataclass
class LabelStudioConfig:
    """Configuration for Label Studio transformation."""
    output_dir: str = "data/human_eval"
    random_seed: int = 42
    include_reference: bool = True
    shuffle_summaries: bool = True


@dataclass
class ResultsConfig:
    """Configuration for results processing."""
    output_dir: str = "data/human_eval"
    expected_rank_count: int = 11
    quality_prefixes: List[str] = field(default_factory=lambda: [
        "hallucination", "incoherent", "publishable"
    ])


class HumanEvalConfigManager:
    """Manages human evaluation configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
    
    def get_data_prep_config(self, overrides: Optional[Dict[str, Any]] = None) -> DataPreparationConfig:
        """Get data preparation configuration."""
        config = self._config.get('data_preparation', {})
        if overrides:
            config.update(overrides.get('data_preparation', {}))
        return DataPreparationConfig(**config)
    
    def get_model_summary_config(self, overrides: Optional[Dict[str, Any]] = None) -> ModelSummaryConfig:
        """Get model summary configuration."""
        config = self._config.get('model_summaries', {})
        if overrides:
            config.update(overrides.get('model_summaries', {}))
        
        # Create config object
        model_config = ModelSummaryConfig(
            output_dir=config.get('output_dir', 'data/human_eval')
        )
        
        # Add model specs
        for spec in config.get('model_specs', []):
            model_config.add_model_spec(**spec)
        
        return model_config
    
    def get_label_studio_config(self, overrides: Optional[Dict[str, Any]] = None) -> LabelStudioConfig:
        """Get Label Studio configuration."""
        config = self._config.get('label_studio', {})
        if overrides:
            config.update(overrides.get('label_studio', {}))
        return LabelStudioConfig(**config)
    
    def get_results_config(self, overrides: Optional[Dict[str, Any]] = None) -> ResultsConfig:
        """Get results processing configuration."""
        config = self._config.get('results', {})
        if overrides:
            config.update(overrides.get('results', {}))
        return ResultsConfig(**config)
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "data_preparation": {
                "input_csv_path": "data/processed/cases_prs_synth_prompts_test_subset.csv",
                "output_dir": "data/human_eval",
                "sample_size_per_court": 10,
                "random_seed": 42,
                "required_columns": ["id", "subset_name", "synthetic_prompt", "judgement", "summary"]
            },
            "model_summaries": {
                "output_dir": "data/human_eval",
                "model_specs": [
                    {
                        "csv_path": "data/generation/full/cases_prs_synth_prompts_test_sample_generated_judgement_summ_llama_3_3_70B.csv",
                        "column_name": "llama_3_3_70B_generated_judgement_summary",
                        "category": "full"
                    },
                    {
                        "csv_path": "data/generation/hier/cases_prs_synth_prompts_test_sample_hier_gen_summ_llama_3_3_70B.csv",
                        "column_name": "llama_3_3_70B_gen_hier_summary",
                        "category": "hierarchical"
                    }
                ]
            },
            "label_studio": {
                "output_dir": "data/human_eval",
                "random_seed": 42,
                "include_reference": True,
                "shuffle_summaries": True
            },
            "results": {
                "output_dir": "data/human_eval",
                "expected_rank_count": 11,
                "quality_prefixes": ["hallucination", "incoherent", "publishable"]
            }
        }