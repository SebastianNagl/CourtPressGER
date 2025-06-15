"""Command-line interface for Teuken fine-tuning."""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional

from .config import TeukenConfigManager
from .dataset import TeukenDatasetPreparer
from .train import TeukenTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeukenCLI:
    """Command-line interface for Teuken hierarchical summarization fine-tuning."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Teuken hierarchical summarization fine-tuning",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global arguments
        parser.add_argument(
            "--config", "-c",
            type=str,
            help="Path to configuration file"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Prepare dataset command
        prepare_parser = subparsers.add_parser(
            "prepare-dataset",
            help="Prepare dataset for fine-tuning"
        )
        self._add_prepare_args(prepare_parser)
        
        # Train command
        train_parser = subparsers.add_parser(
            "train",
            help="Train Teuken model"
        )
        self._add_train_args(train_parser)
        
        # Create config command
        config_parser = subparsers.add_parser(
            "create-config",
            help="Create a default configuration file"
        )
        config_parser.add_argument(
            "--output", "-o",
            type=str,
            default="teuken_config.json",
            help="Output path for configuration file"
        )
        
        return parser
    
    def _add_prepare_args(self, parser: argparse.ArgumentParser) -> None:
        """Add dataset preparation arguments."""
        parser.add_argument(
            "--input", "-i",
            type=str,
            required=True,
            help="Input CSV file path"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            default="data/processed",
            help="Output directory for processed data"
        )
        parser.add_argument(
            "--train-split", "-t",
            type=float,
            default=0.9,
            help="Training data split ratio (0-1)"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for splitting"
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default="openGPT-X/Teuken-7B-instruct-research-v0.4",
            help="Tokenizer model name"
        )
    
    def _add_train_args(self, parser: argparse.ArgumentParser) -> None:
        """Add training arguments."""
        parser.add_argument(
            "--model",
            type=str,
            help="Model name or path"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for checkpoints"
        )
        parser.add_argument(
            "--train-data",
            type=str,
            help="Training data path"
        )
        parser.add_argument(
            "--val-data",
            type=str,
            help="Validation data path"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of training epochs"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Per-device training batch size"
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            help="Learning rate"
        )
        parser.add_argument(
            "--resume",
            type=str,
            help="Resume from checkpoint (path or 'LAST')"
        )
        parser.add_argument(
            "--push-to-hub",
            action="store_true",
            help="Push model to Hugging Face Hub"
        )
        parser.add_argument(
            "--hub-repo-id",
            type=str,
            help="Hugging Face Hub repository ID"
        )
        parser.add_argument(
            "--wandb-project",
            type=str,
            default="teuken-hier",
            help="Weights & Biases project name"
        )
    
    def _extract_overrides(self, args: argparse.Namespace) -> dict:
        """Extract configuration overrides from command-line arguments."""
        overrides = {
            'model': {},
            'training': {},
            'data': {},
            'hub': {}
        }
        
        # Model overrides
        if hasattr(args, 'model') and args.model:
            overrides['model']['model_name'] = args.model
        
        # Training overrides
        if hasattr(args, 'output_dir') and args.output_dir:
            overrides['training']['output_dir'] = args.output_dir
        if hasattr(args, 'epochs') and args.epochs:
            overrides['training']['num_train_epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            overrides['training']['per_device_train_batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            overrides['training']['learning_rate'] = args.learning_rate
        
        # Data overrides
        if hasattr(args, 'train_data') and args.train_data:
            overrides['data']['train_data_path'] = args.train_data
        if hasattr(args, 'val_data') and args.val_data:
            overrides['data']['val_data_path'] = args.val_data
        
        # Hub overrides
        if hasattr(args, 'push_to_hub'):
            overrides['hub']['push_to_hub'] = args.push_to_hub
        if hasattr(args, 'hub_repo_id') and args.hub_repo_id:
            overrides['hub']['hub_repo_id'] = args.hub_repo_id
        
        return overrides
    
    def run(self, argv: Optional[list] = None) -> int:
        """Run the CLI."""
        args = self.parser.parse_args(argv)
        
        if not args.command:
            self.parser.print_help()
            return 1
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            if args.command == "create-config":
                # Create default configuration
                config_manager = TeukenConfigManager()
                config = config_manager.create_default_config()
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Created default configuration file: {args.output}")
                return 0
            
            elif args.command == "prepare-dataset":
                # Prepare dataset
                preparer = TeukenDatasetPreparer(
                    tokenizer_name=args.tokenizer,
                    seed=args.seed
                )
                
                preparer.prepare_dataset(
                    input_path=args.input,
                    output_dir=args.output_dir,
                    train_split=args.train_split
                )
                
                return 0
            
            elif args.command == "train":
                # Load configuration
                if args.config:
                    config_manager = TeukenConfigManager(args.config)
                else:
                    config_manager = TeukenConfigManager()
                    config_manager._config = config_manager.create_default_config()
                
                # Apply overrides
                overrides = self._extract_overrides(args)
                
                # Get configurations
                model_config = config_manager.get_model_config(overrides)
                training_config = config_manager.get_training_config(overrides)
                data_config = config_manager.get_data_config(overrides)
                hub_config = config_manager.get_hub_config(overrides)
                
                # Set environment variables
                if hasattr(args, 'wandb_project'):
                    import os
                    os.environ['WANDB_PROJECT'] = args.wandb_project
                
                # Create trainer
                trainer = TeukenTrainer(
                    model_config=model_config,
                    training_config=training_config,
                    data_config=data_config,
                    hub_config=hub_config
                )
                
                # Handle resume
                resume_from = None
                if hasattr(args, 'resume') and args.resume:
                    from .utils import resolve_checkpoint
                    resume_from = resolve_checkpoint(
                        args.resume, 
                        Path(training_config.output_dir)
                    )
                
                # Train
                trainer.train(resume_from_checkpoint=resume_from)
                
                return 0
            
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    cli = TeukenCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()