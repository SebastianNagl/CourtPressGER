"""Training module for Teuken fine-tuning."""

import os
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback
)
from trl import SFTConfig, SFTTrainer

from .config import (
    TeukenModelConfig,
    TeukenTrainingConfig,
    TeukenDataConfig,
    HubConfig
)
from .utils import (
    GradNormWandBCallback,
    resolve_checkpoint,
    init_or_resume_wandb,
    push_to_hf_hub
)

logger = logging.getLogger(__name__)


class TeukenTrainer:
    """Handles training of Teuken models."""
    
    def __init__(self,
                 model_config: TeukenModelConfig,
                 training_config: TeukenTrainingConfig,
                 data_config: TeukenDataConfig,
                 hub_config: HubConfig):
        """
        Initialize Teuken trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            hub_config: Hub configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.hub_config = hub_config
        
        # Set seed
        self._set_seed(training_config.seed)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            use_fast=False,
            trust_remote_code=True
        )
    
    def _load_model(self):
        """Load model."""
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(self.model_config.torch_dtype, torch.bfloat16)
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True
        }
        
        if self.model_config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            **model_kwargs
        )
        
        # Configure model
        if self.model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.config.use_cache = False
        self.model.config.dropout = self.model_config.dropout
    
    def _load_datasets(self):
        """Load training and validation datasets."""
        logger.info("Loading datasets")
        
        train_ds = load_dataset(
            "json",
            data_files=self.data_config.train_data_path,
            split="train"
        )
        
        val_ds = load_dataset(
            "json",
            data_files=self.data_config.val_data_path,
            split="train"
        )
        
        logger.info(f"Train dataset size: {len(train_ds)}")
        logger.info(f"Validation dataset size: {len(val_ds)}")
        
        return train_ds, val_ds
    
    def _create_trainer_config(self) -> SFTConfig:
        """Create SFT trainer configuration."""
        return SFTConfig(
            seed=self.training_config.seed,
            output_dir=self.training_config.output_dir,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            num_train_epochs=self.training_config.num_train_epochs,
            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_steps=self.training_config.warmup_steps,
            weight_decay=self.training_config.weight_decay,
            max_length=self.training_config.max_length,
            packing=self.data_config.packing,
            logging_steps=self.training_config.logging_steps,
            completion_only_loss=self.data_config.completion_only_loss,
            eval_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            save_strategy="steps",
            save_total_limit=2,
            save_steps=self.training_config.save_steps,
            per_device_eval_batch_size=1,
            dataset_text_field=self.data_config.dataset_text_field,
            bf16=self.training_config.bf16,
            optim=self.training_config.optim,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=False,
            eval_accumulation_steps=1,
            run_name=f"teuken-hier-sft-{self.training_config.seed}",
            report_to=["wandb"],
            remove_unused_columns=False,
            max_grad_norm=self.training_config.max_grad_norm,
        )
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Run training.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Load components
        self._load_tokenizer()
        self._load_model()
        train_ds, val_ds = self._load_datasets()
        
        # Create trainer config
        trainer_config = self._create_trainer_config()
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=trainer_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            callbacks=[
                GradNormWandBCallback(),
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=0.0
                ),
            ],
        )
        
        # Initialize Weights & Biases
        run_id_file = Path(self.training_config.output_dir) / "wandb_run_id.txt"
        init_or_resume_wandb(
            accelerator=self.trainer.accelerator,
            run_id_file=run_id_file,
            trainer_cfg=trainer_config,
            resume_ckpt=resume_from_checkpoint,
            notes="Teuken-7B SFT - hierarchical summarization",
            project=os.getenv("WANDB_PROJECT", "teuken-hier"),
        )
        
        # Train
        logger.info("Starting training")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        if self.trainer.accelerator.is_local_main_process:
            final_dir = Path(self.training_config.output_dir) / "best"
            logger.info(f"Saving final model to {final_dir}")
            self.trainer.save_model(final_dir)
            self.tokenizer.save_pretrained(final_dir)
            
            # Push to hub if configured
            if self.hub_config.push_to_hub:
                hf_token = os.getenv("HF_API_KEY")
                if hf_token:
                    push_to_hf_hub(
                        full_dir=final_dir,
                        repo_id=self.hub_config.hub_repo_id,
                        hf_token=hf_token,
                        private=self.hub_config.hub_private
                    )
                else:
                    logger.warning("HF_API_KEY not found, skipping hub upload")
            
            # Finish Weights & Biases
            import wandb
            wandb.finish()