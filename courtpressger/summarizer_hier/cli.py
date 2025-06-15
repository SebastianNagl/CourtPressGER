"""Command-line interface for hierarchical summarization."""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .config import ConfigManager, create_default_config
from .text_chunker import run_chunking
from .hierarchical_summarizer import run_summarization
from .summary_generator import run_generation
from .summary_aggregator import run_aggregation
from .multi_gpu_processor import run_multi_gpu_processing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for hierarchical summarization."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            description="Hierarchical summarization toolkit for court documents",
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
        
        # Chunk command
        chunk_parser = subparsers.add_parser(
            "chunk",
            help="Chunk documents into smaller segments"
        )
        self._add_chunking_args(chunk_parser)
        
        # Summarize command
        summarize_parser = subparsers.add_parser(
            "summarize",
            help="Generate hierarchical summaries"
        )
        self._add_summarization_args(summarize_parser)
        
        # Generate command
        generate_parser = subparsers.add_parser(
            "generate",
            help="Generate final summaries from hierarchical summaries"
        )
        self._add_generation_args(generate_parser)
        
        # Aggregate command
        aggregate_parser = subparsers.add_parser(
            "aggregate",
            help="Aggregate summaries from multiple files"
        )
        self._add_aggregation_args(aggregate_parser)
        
        # Config command
        config_parser = subparsers.add_parser(
            "create-config",
            help="Create a default configuration file"
        )
        config_parser.add_argument(
            "--output", "-o",
            type=str,
            default="config.json",
            help="Output path for configuration file"
        )
        
        return parser
    
    def _add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common arguments to a subparser."""
        parser.add_argument("--input", "-i", type=str, help="Input file path")
        parser.add_argument("--output", "-o", type=str, help="Output file path")
        parser.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU processing")
        parser.add_argument("--gpu-count", type=int, help="Number of GPUs to use")
    
    def _add_chunking_args(self, parser: argparse.ArgumentParser) -> None:
        """Add chunking-specific arguments."""
        self._add_common_args(parser)
        parser.add_argument("--chunk-size", type=int, help="Maximum tokens per chunk")
        parser.add_argument("--overlap", type=int, help="Token overlap between chunks")
    
    def _add_summarization_args(self, parser: argparse.ArgumentParser) -> None:
        """Add summarization-specific arguments."""
        self._add_common_args(parser)
        parser.add_argument("--max-context", type=int, help="Maximum context length")
        parser.add_argument("--max-summary", type=int, help="Maximum summary length")
        parser.add_argument("--validate", action="store_true", help="Validate summary quality")
        parser.add_argument("--prompts-dir", type=str, help="Directory containing prompt templates")
    
    def _add_generation_args(self, parser: argparse.ArgumentParser) -> None:
        """Add generation-specific arguments."""
        self._add_common_args(parser)
        parser.add_argument("--temperature", type=float, help="Generation temperature")
        parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    
    def _add_aggregation_args(self, parser: argparse.ArgumentParser) -> None:
        """Add aggregation-specific arguments."""
        parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory with summaries")
        parser.add_argument("--output", "-o", type=str, required=True, help="Output file path")
        parser.add_argument("--original", type=str, help="Original dataset for merging")
    
    def _prepare_config(self, args: argparse.Namespace) -> ConfigManager:
        """Prepare configuration from file and command-line overrides."""
        # Load base configuration
        if args.config:
            config_manager = ConfigManager(args.config)
        else:
            # Create default configuration
            config_manager = ConfigManager()
            config_manager._config = create_default_config()
        
        # Apply command-line overrides
        overrides = self._extract_overrides(args)
        
        return config_manager, overrides
    
    def _extract_overrides(self, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        """Extract configuration overrides from command-line arguments."""
        overrides = {
            'model': {},
            'chunking': {},
            'summarization': {},
            'processing': {}
        }
        
        # Common overrides
        if hasattr(args, 'input') and args.input:
            overrides['processing']['input_path'] = args.input
        if hasattr(args, 'output') and args.output:
            overrides['processing']['output_path'] = args.output
        if hasattr(args, 'gpu_count') and args.gpu_count:
            overrides['processing']['gpu_count'] = args.gpu_count
        
        # Chunking overrides
        if hasattr(args, 'chunk_size') and args.chunk_size:
            overrides['chunking']['chunk_size'] = args.chunk_size
        if hasattr(args, 'overlap') and args.overlap:
            overrides['chunking']['overlap'] = args.overlap
        
        # Summarization overrides
        if hasattr(args, 'max_context') and args.max_context:
            overrides['summarization']['max_context_length'] = args.max_context
        if hasattr(args, 'max_summary') and args.max_summary:
            overrides['summarization']['max_summary_length'] = args.max_summary
        if hasattr(args, 'validate') and args.validate:
            overrides['summarization']['validate_summary'] = args.validate
        if hasattr(args, 'prompts_dir') and args.prompts_dir:
            overrides['summarization']['prompts_dir'] = args.prompts_dir
        
        # Generation overrides
        if hasattr(args, 'temperature') and args.temperature:
            overrides['model']['temperature'] = args.temperature
        if hasattr(args, 'max_tokens') and args.max_tokens:
            overrides['summarization']['max_summary_length'] = args.max_tokens
        
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
                # Create default configuration file
                config = create_default_config()
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Created default configuration file: {args.output}")
                return 0
            
            # Prepare configuration
            config_manager, overrides = self._prepare_config(args)
            
            # Validate paths
            config_manager.validate_paths()
            
            # Execute command
            if args.command == "chunk":
                if args.multi_gpu:
                    run_multi_gpu_processing(
                        task="chunk",
                        config_manager=config_manager,
                        overrides=overrides
                    )
                else:
                    run_chunking(config_manager, overrides)
            
            elif args.command == "summarize":
                if args.multi_gpu:
                    run_multi_gpu_processing(
                        task="summarize",
                        config_manager=config_manager,
                        overrides=overrides
                    )
                else:
                    run_summarization(config_manager, overrides)
            
            elif args.command == "generate":
                if args.multi_gpu:
                    run_multi_gpu_processing(
                        task="generate",
                        config_manager=config_manager,
                        overrides=overrides
                    )
                else:
                    run_generation(config_manager, overrides)
            
            elif args.command == "aggregate":
                # Special case for aggregation
                run_aggregation(
                    input_dir=args.input_dir,
                    output_path=args.output,
                    original_path=args.original
                )
            
            return 0
            
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    cli = CLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()