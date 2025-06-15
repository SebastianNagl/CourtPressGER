"""Utility functions for Teuken training."""

import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict

import torch
import wandb
from accelerate import Accelerator
from transformers import TrainerCallback
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


class GradNormWandBCallback(TrainerCallback):
    """Logs gradient norm to Weights & Biases."""
    
    def on_train_begin(self, *_, **__):
        self.last_grad_norm = None
    
    def _compute_grad_norm(self, trainer, model) -> torch.Tensor:
        """Compute gradient norm."""
        if trainer.deepspeed:
            return trainer.deepspeed.get_grad_norm()
        
        grads = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return torch.norm(torch.stack(grads))
    
    def on_backward_end(self, args, state, control, **kw):
        """Log gradient norm after backward pass."""
        trainer, model = kw["trainer"], kw["model"]
        g = self._compute_grad_norm(trainer, model)
        g = trainer.accelerator.gather_for_metrics(g).mean()
        self.last_grad_norm = g.item()
        
        if trainer.accelerator.is_local_main_process:
            trainer.log({"grad_norm": self.last_grad_norm})
    
    def on_log(self, args, state, control, logs=None, **kw):
        """Add gradient norm to logs."""
        if logs is not None and self.last_grad_norm is not None:
            logs["grad_norm"] = self.last_grad_norm


def resolve_checkpoint(checkpoint_path: str, output_dir: Path) -> Optional[str]:
    """
    Resolve checkpoint path.
    
    Args:
        checkpoint_path: Checkpoint specification ('LAST' or path)
        output_dir: Output directory to search for checkpoints
        
    Returns:
        Resolved checkpoint path or None
    """
    if not checkpoint_path or checkpoint_path.lower() == "none":
        return None
    
    if checkpoint_path.upper() == "LAST":
        # Find latest checkpoint
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoints:
            logger.info(f"Found latest checkpoint: {checkpoints[-1]}")
            return str(checkpoints[-1])
        else:
            logger.warning("No checkpoints found")
            return None
    
    # Treat as literal path
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    return str(checkpoint_path)


def init_or_resume_wandb(
    accelerator: Accelerator,
    run_id_file: Path,
    trainer_cfg: Any,
    resume_ckpt: Optional[str],
    notes: str = "",
    project: str = "teuken-runs",
) -> None:
    """
    Initialize or resume Weights & Biases run.
    
    Args:
        accelerator: Accelerator instance
        run_id_file: File to store run ID
        trainer_cfg: Trainer configuration
        resume_ckpt: Checkpoint path if resuming
        notes: Notes for the run
        project: W&B project name
    """
    if not accelerator.is_local_main_process:
        return
    
    # Determine run ID
    if resume_ckpt and run_id_file.exists():
        run_id = run_id_file.read_text().strip()
        logger.info(f"Resuming W&B run: {run_id}")
    else:
        run_id = wandb.util.generate_id()
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(run_id)
        logger.info(f"Starting new W&B run: {run_id}")
    
    # Initialize W&B
    wandb.init(
        project=project,
        name=trainer_cfg.run_name,
        id=run_id,
        resume="allow",
        notes=notes,
        config=(
            trainer_cfg.to_dict()
            if hasattr(trainer_cfg, "to_dict")
            else {k: v for k, v in vars(trainer_cfg).items() if not k.startswith("_")}
        ),
    )


def push_to_hf_hub(
    full_dir: Path,
    repo_id: str,
    hf_token: str,
    commit_message: str = "Upload fine-tuned checkpoint",
    private: bool = False,
) -> None:
    """
    Push model to Hugging Face Hub.
    
    Args:
        full_dir: Directory containing the model
        repo_id: Repository ID on HF Hub
        hf_token: HF API token
        commit_message: Commit message
        private: Whether to make repo private
    """
    logger.info(f"Pushing model to {repo_id}")
    
    api = HfApi(token=hf_token)
    
    # Create repository if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )
    
    # Upload folder
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(full_dir),
        path_in_repo=".",
        commit_message=commit_message,
    )
    
    logger.info(f"Successfully pushed to https://huggingface.co/{repo_id}")