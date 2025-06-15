"""Configuration management for Teuken fine-tuning."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TeukenModelConfig:
    """Teuken model configuration."""
    model_name: str = "openGPT-X/Teuken-7B-instruct-research-v0.4"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    dropout: float = 0.1


@dataclass
class TeukenTrainingConfig:
    """Training configuration for Teuken fine-tuning."""
    output_dir: str = "teuken-hier-sft"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 128
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_length: int = 4096
    eval_steps: int = 25
    save_steps: int = 25
    logging_steps: int = 2
    seed: int = 42
    bf16: bool = True
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 1.0
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    early_stopping_patience: int = 4


@dataclass
class TeukenDataConfig:
    """Data configuration for Teuken fine-tuning."""
    train_data_path: str = "data/processed/train.jsonl"
    val_data_path: str = "data/processed/validation.jsonl"
    dataset_text_field: str = "prompt"
    completion_only_loss: bool = True
    packing: bool = False


@dataclass
class HubConfig:
    """Hugging Face Hub configuration."""
    push_to_hub: bool = True
    hub_repo_id: str = "teuken-hier-summ-sft"
    hub_private: bool = False


class TeukenConfigManager:
    """Manages Teuken training configuration."""
    
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
    
    def get_model_config(self, overrides: Optional[Dict[str, Any]] = None) -> TeukenModelConfig:
        """Get model configuration."""
        config = self._config.get('model', {})
        if overrides:
            config.update(overrides.get('model', {}))
        return TeukenModelConfig(**config)
    
    def get_training_config(self, overrides: Optional[Dict[str, Any]] = None) -> TeukenTrainingConfig:
        """Get training configuration."""
        config = self._config.get('training', {})
        if overrides:
            config.update(overrides.get('training', {}))
        return TeukenTrainingConfig(**config)
    
    def get_data_config(self, overrides: Optional[Dict[str, Any]] = None) -> TeukenDataConfig:
        """Get data configuration."""
        config = self._config.get('data', {})
        if overrides:
            config.update(overrides.get('data', {}))
        return TeukenDataConfig(**config)
    
    def get_hub_config(self, overrides: Optional[Dict[str, Any]] = None) -> HubConfig:
        """Get hub configuration."""
        config = self._config.get('hub', {})
        if overrides:
            config.update(overrides.get('hub', {}))
        return HubConfig(**config)
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "model": {
                "model_name": "openGPT-X/Teuken-7B-instruct-research-v0.4",
                "torch_dtype": "bfloat16",
                "use_flash_attention": True,
                "gradient_checkpointing": False,
                "dropout": 0.1
            },
            "training": {
                "output_dir": "teuken-hier-sft",
                "num_train_epochs": 4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 128,
                "learning_rate": 2e-5,
                "lr_scheduler_type": "constant",
                "warmup_steps": 0,
                "weight_decay": 0.01,
                "max_length": 4096,
                "eval_steps": 25,
                "save_steps": 25,
                "logging_steps": 2,
                "seed": 42,
                "bf16": True,
                "optim": "adamw_torch_fused",
                "max_grad_norm": 1.0,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "early_stopping_patience": 4
            },
            "data": {
                "train_data_path": "data/processed/train.jsonl",
                "val_data_path": "data/processed/test.jsonl",
                "dataset_text_field": "prompt",
                "completion_only_loss": True,
                "packing": False
            },
            "hub": {
                "push_to_hub": True,
                "hub_repo_id": "teuken-hier-summ-sft",
                "hub_private": False
            }
        }