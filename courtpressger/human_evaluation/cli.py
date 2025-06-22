"""Command-line interface for human evaluation."""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional

from .config import HumanEvalConfigManager
from .data_preparation import HumanEvalDataPreparer
from .label_studio import LabelStudioTransformer
from .results_processor import ResultsProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HumanEvalCLI:
    """Command-line interface for human evaluation workflow."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Human evaluation workflow for court press releases",
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
        
        # Prepare data command
        prepare_parser = subparsers.add_parser(
            "prepare",
            help="Prepare evaluation dataset with sampling"
        )
        self._add_prepare_args(prepare_parser)
        
        # Augment with model summaries command
        augment_parser = subparsers.add_parser(
            "augment",
            help="Augment dataset with model-generated summaries"
        )
        self._add_augment_args(augment_parser)
        
        # Transform for Label Studio command
        transform_parser = subparsers.add_parser(
            "transform",
            help="Transform data for Label Studio import"
        )
        self._add_transform_args(transform_parser)
        
        # Process results command
        process_parser = subparsers.add_parser(
            "process-results",
            help="Process Label Studio annotation results"
        )
        self._add_process_args(process_parser)
        
        # Full pipeline command
        pipeline_parser = subparsers.add_parser(
            "pipeline",
            help="Run full evaluation pipeline"
        )
        self._add_pipeline_args(pipeline_parser)
        
        # Create config command
        config_parser = subparsers.add_parser(
            "create-config",
            help="Create default configuration file"
        )
        config_parser.add_argument(
            "--output", "-o",
            type=str,
            default="human_eval_config.json",
            help="Output path for configuration file"
        )
        
        return parser
    
    def _add_prepare_args(self, parser: argparse.ArgumentParser) -> None:
        """Add data preparation arguments."""
        parser.add_argument(
            "--input", "-i",
            type=str,
            help="Input CSV file path"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory"
        )
        parser.add_argument(
            "--sample-size",
            type=int,
            help="Sample size per court"
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed"
        )
    
    def _add_augment_args(self, parser: argparse.ArgumentParser) -> None:
        """Add augmentation arguments."""
        parser.add_argument(
            "--input", "-i",
            type=str,
            help="Input JSON file path"
        )
        parser.add_argument(
            "--output", "-o",
            type=str,
            help="Output JSON file path"
        )
        parser.add_argument(
            "--model-specs",
            type=str,
            help="Path to model specifications JSON"
        )
    
    def _add_transform_args(self, parser: argparse.ArgumentParser) -> None:
        """Add Label Studio transformation arguments."""
        parser.add_argument(
            "--input", "-i",
            type=str,
            help="Input JSON file path"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory"
        )
        parser.add_argument(
            "--no-shuffle",
            action="store_true",
            help="Don't shuffle summaries"
        )
    
    def _add_process_args(self, parser: argparse.ArgumentParser) -> None:
        """Add results processing arguments."""
        parser.add_argument(
            "--results", "-r",
            type=str,
            required=True,
            help="Label Studio results JSON file"
        )
        parser.add_argument(
            "--mapping", "-m",
            type=str,
            required=True,
            help="Press-to-model mapping JSON file"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory"
        )
    
    def _add_pipeline_args(self, parser: argparse.ArgumentParser) -> None:
        """Add full pipeline arguments."""
        parser.add_argument(
            "--input", "-i",
            type=str,
            help="Input CSV file path"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for all files"
        )
        parser.add_argument(
            "--skip-prepare",
            action="store_true",
            help="Skip data preparation step"
        )
        parser.add_argument(
            "--skip-augment",
            action="store_true",
            help="Skip augmentation step"
        )
        parser.add_argument(
            "--skip-transform",
            action="store_true",
            help="Skip Label Studio transformation"
        )
    
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
            # Load configuration
            if args.config:
                config_manager = HumanEvalConfigManager(args.config)
            else:
                config_manager = HumanEvalConfigManager()
                config_manager._config = config_manager.create_default_config()
            
            if args.command == "create-config":
                # Create default configuration
                config = config_manager.create_default_config()
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Created default configuration: {args.output}")
                return 0
            
            elif args.command == "prepare":
                # Prepare evaluation dataset
                overrides = self._extract_prepare_overrides(args)
                config = config_manager.get_data_prep_config(overrides)
                
                preparer = HumanEvalDataPreparer(config)
                preparer.prepare_evaluation_dataset()
                return 0
            
            elif args.command == "augment":
                # Augment with model summaries
                overrides = self._extract_augment_overrides(args)
                model_config = config_manager.get_model_summary_config(overrides)
                
                preparer = HumanEvalDataPreparer(config_manager.get_data_prep_config())
                preparer.augment_with_model_summaries(
                    input_path=args.input,
                    output_path=args.output,
                    model_config=model_config
                )
                return 0
            
            elif args.command == "transform":
                # Transform for Label Studio
                overrides = self._extract_transform_overrides(args)
                config = config_manager.get_label_studio_config(overrides)
                
                transformer = LabelStudioTransformer(config)
                transformer.transform_for_label_studio(
                    input_path=args.input,
                    output_dir=args.output_dir or config.output_dir
                )
                return 0
            
            elif args.command == "process-results":
                # Process Label Studio results
                overrides = self._extract_process_overrides(args)
                config = config_manager.get_results_config(overrides)
                
                processor = ResultsProcessor(config)
                processor.process_results(
                    results_path=args.results,
                    mapping_path=args.mapping,
                    output_dir=args.output_dir or config.output_dir
                )
                return 0
            
            elif args.command == "pipeline":
                # Run full pipeline
                return self._run_pipeline(args, config_manager)
            
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return 1
    
    def _extract_prepare_overrides(self, args) -> dict:
        """Extract overrides for data preparation."""
        overrides = {'data_preparation': {}}
        
        if hasattr(args, 'input') and args.input:
            overrides['data_preparation']['input_csv_path'] = args.input
        if hasattr(args, 'output_dir') and args.output_dir:
            overrides['data_preparation']['output_dir'] = args.output_dir
        if hasattr(args, 'sample_size') and args.sample_size:
            overrides['data_preparation']['sample_size_per_court'] = args.sample_size
        if hasattr(args, 'seed') and args.seed:
            overrides['data_preparation']['random_seed'] = args.seed
        
        return overrides
    
    def _extract_augment_overrides(self, args) -> dict:
        """Extract overrides for augmentation."""
        overrides = {'model_summaries': {}}
        
        if hasattr(args, 'model_specs') and args.model_specs:
            # Load model specs from file
            with open(args.model_specs, 'r') as f:
                specs = json.load(f)
            overrides['model_summaries']['model_specs'] = specs
        
        return overrides
    
    def _extract_transform_overrides(self, args) -> dict:
        """Extract overrides for Label Studio transformation."""
        overrides = {'label_studio': {}}
        
        if hasattr(args, 'no_shuffle') and args.no_shuffle:
            overrides['label_studio']['shuffle_summaries'] = False
        
        return overrides
    
    def _extract_process_overrides(self, args) -> dict:
        """Extract overrides for results processing."""
        return {'results': {}}
    
    def _run_pipeline(self, args, config_manager) -> int:
        """Run the full evaluation pipeline."""
        output_dir = Path(args.output_dir or "data/human_eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Prepare data
        if not args.skip_prepare:
            logger.info("Step 1: Preparing evaluation dataset...")
            data_config = config_manager.get_data_prep_config({
                'data_preparation': {
                    'input_csv_path': args.input,
                    'output_dir': str(output_dir)
                }
            })
            preparer = HumanEvalDataPreparer(data_config)
            base_json = preparer.prepare_evaluation_dataset()
        else:
            base_json = output_dir / "human_eval_base.json"
        
        # Step 2: Augment with model summaries
        if not args.skip_augment:
            logger.info("Step 2: Augmenting with model summaries...")
            model_config = config_manager.get_model_summary_config()
            preparer = HumanEvalDataPreparer(config_manager.get_data_prep_config())
            augmented_json = preparer.augment_with_model_summaries(
                input_path=str(base_json),
                output_path=str(output_dir / "human_eval_augmented.json"),
                model_config=model_config
            )
        else:
            augmented_json = output_dir / "human_eval_augmented.json"
        
        # Step 3: Transform for Label Studio
        if not args.skip_transform:
            logger.info("Step 3: Transforming for Label Studio...")
            ls_config = config_manager.get_label_studio_config()
            transformer = LabelStudioTransformer(ls_config)
            transformer.transform_for_label_studio(
                input_path=str(augmented_json),
                output_dir=str(output_dir)
            )
        
        logger.info("Pipeline complete!")
        logger.info(f"Output files in: {output_dir}")
        return 0


def main():
    """Main entry point."""
    cli = HumanEvalCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()