"""Configuration management for hierarchical summarization."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_type: str
    model_name: str
    tokenizer_name: Optional[str] = None
    device: str = "cuda:0"
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True
    torch_dtype: str = "bfloat16"
    api_key_env: Optional[str] = None


@dataclass
class ChunkConfig:
    """Chunking configuration settings."""
    chunk_size: int = 2048
    overlap: int = 0


@dataclass
class SummarizationConfig:
    """Summarization configuration settings."""
    max_context_length: int = 4096
    max_summary_length: int = 900
    chunk_size: int = 2048
    validate_summary: bool = False
    num_attempts: int = 3
    word_ratio: float = 0.65
    temperature: float = 0.1
    prompts_dir: str = "prompts"


@dataclass
class ProcessingConfig:
    """General processing configuration."""
    input_path: str
    output_path: str
    batch_size: int = 10
    checkpoint_frequency: int = 50
    gpu_count: int = 1
    column_mappings: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading and validation."""
    
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
    
    def get_model_config(self, override: Optional[Dict[str, Any]] = None) -> ModelConfig:
        """Get model configuration with optional overrides."""
        config = self._config.get('model', {})
        if override:
            config.update(override)
        return ModelConfig(**config)
    
    def get_chunk_config(self, override: Optional[Dict[str, Any]] = None) -> ChunkConfig:
        """Get chunking configuration with optional overrides."""
        config = self._config.get('chunking', {})
        if override:
            config.update(override)
        return ChunkConfig(**config)
    
    def get_summarization_config(self, override: Optional[Dict[str, Any]] = None) -> SummarizationConfig:
        """Get summarization configuration with optional overrides."""
        config = self._config.get('summarization', {})
        if override:
            config.update(override)
        return SummarizationConfig(**config)
    
    def get_processing_config(self, override: Optional[Dict[str, Any]] = None) -> ProcessingConfig:
        """Get processing configuration with optional overrides."""
        config = self._config.get('processing', {})
        if override:
            config.update(override)
        
        # Set default column mappings
        if 'column_mappings' not in config:
            config['column_mappings'] = {
                'judgement_column': 'judgement',
                'prompt_column': 'synthetic_prompt',
                'summary_column': 'summary',
                'chunks_column': 'chunks',
                'hierarchical_summary_column': 'hierarchical_summary',
                'generated_summary_column': 'generated_summary'
            }
        
        return ProcessingConfig(**config)
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist."""
        processing = self.get_processing_config()
        
        if not Path(processing.input_path).exists():
            raise FileNotFoundError(f"Input path does not exist: {processing.input_path}")
        
        output_dir = Path(processing.output_path).parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return True


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration template."""
    return {
        "model": {
            "model_type": "huggingface_local",
            "model_name": "openGPT-X/Teuken-7B-instruct-research-v0.4",
            "tokenizer_name": "openGPT-X/Teuken-7B-instruct-research-v0.4",
            "device": "cuda:0",
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "torch_dtype": "bfloat16"
        },
        "chunking": {
            "chunk_size": 2048,
            "overlap": 0
        },
        "summarization": {
            "max_context_length": 4096,
            "max_summary_length": 900,
            "chunk_size": 2048,
            "validate_summary": False,
            "num_attempts": 3,
            "word_ratio": 0.65,
            "temperature": 0.1,
            "prompts_dir": "prompts"
        },
        "processing": {
            "input_path": "data/input.csv",
            "output_path": "data/output.csv",
            "batch_size": 10,
            "checkpoint_frequency": 50,
            "gpu_count": 1,
            "column_mappings": {
                "judgement_column": "judgement",
                "prompt_column": "synthetic_prompt",
                "summary_column": "summary",
                "chunks_column": "chunks",
                "hierarchical_summary_column": "hierarchical_summary",
                "generated_summary_column": "generated_summary"
            }
        }
    }