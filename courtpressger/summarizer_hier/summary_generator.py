"""Module for generating final summaries from hierarchical summaries."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from .config import ConfigManager
from .model_interface import get_model_interface

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Generates final summaries using hierarchical summaries and prompts."""
    
    def __init__(self, model_interface, config: dict):
        """
        Initialize the summary generator.
        
        Args:
            model_interface: Model interface for text generation
            config: Generation configuration
        """
        self.model_interface = model_interface
        self.config = config
        self.max_context_length = config.get('max_context_length', 4096)
        self.max_summary_length = config.get('max_summary_length', 1024)
        self.temperature = config.get('temperature', 0.1)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.model_interface.count_tokens(text)
    
    def generate_summary(self, prompt: str, hierarchical_summary: str) -> str:
        """
        Generate a final summary using prompt and hierarchical summary.
        
        Args:
            prompt: The prompt to use
            hierarchical_summary: The hierarchical summary
            
        Returns:
            Generated summary
        """
        # Combine prompt and hierarchical summary
        full_prompt = f"{prompt}\n\n{hierarchical_summary}"
        
        # Check if it fits within context
        if self.count_tokens(full_prompt) > self.max_context_length - self.max_summary_length:
            # Truncate hierarchical summary if needed
            available_tokens = self.max_context_length - self.max_summary_length - self.count_tokens(prompt) - 10
            
            # Simple truncation - could be improved
            tokens = self.model_interface.tokenizer.encode(hierarchical_summary)
            truncated_tokens = tokens[:available_tokens]
            hierarchical_summary = self.model_interface.tokenizer.decode(truncated_tokens)
            full_prompt = f"{prompt}\n\n{hierarchical_summary}"
            
            logger.warning("Truncated hierarchical summary to fit context window")
        
        # Generate summary
        messages = [{'role': 'user', 'content': full_prompt}]
        summary = self.model_interface.generate_text(
            messages, 
            self.max_summary_length, 
            self.temperature
        )
        
        return summary
    
    def process_dataframe(self, 
                         df: pd.DataFrame,
                         prompt_column: str,
                         hierarchical_summary_column: str,
                         output_column: str) -> pd.DataFrame:
        """
        Process a dataframe to generate final summaries.
        
        Args:
            df: Input dataframe
            prompt_column: Column containing prompts
            hierarchical_summary_column: Column containing hierarchical summaries
            output_column: Column to store generated summaries
            
        Returns:
            Dataframe with generated summaries
        """
        summaries = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
            prompt = row[prompt_column]
            hierarchical_summary = row[hierarchical_summary_column]
            
            try:
                summary = self.generate_summary(prompt, hierarchical_summary)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error generating summary for row {idx}: {e}")
                summaries.append("")
        
        df[output_column] = summaries
        return df


def run_generation(config_manager: ConfigManager, overrides: dict) -> None:
    """
    Run the summary generation process.
    
    Args:
        config_manager: Configuration manager
        overrides: Configuration overrides
    """
    # Get configurations
    model_config = config_manager.get_model_config(overrides.get('model'))
    processing_config = config_manager.get_processing_config(overrides.get('processing'))
    
    # Extract generation-specific config
    generation_config = {
        'max_context_length': overrides.get('summarization', {}).get('max_context_length', 4096),
        'max_summary_length': overrides.get('summarization', {}).get('max_summary_length', 1024),
        'temperature': overrides.get('model', {}).get('temperature', 0.1)
    }
    
    logger.info("Starting summary generation...")
    logger.info(f"Input: {processing_config.input_path}")
    logger.info(f"Output: {processing_config.output_path}")
    
    # Load data
    df = pd.read_csv(processing_config.input_path)
    
    # Get column names
    prompt_column = processing_config.column_mappings['prompt_column']
    hierarchical_summary_column = processing_config.column_mappings['hierarchical_summary_column']
    output_column = processing_config.column_mappings['generated_summary_column']
    
    # Validate columns exist
    required_columns = [prompt_column, hierarchical_summary_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize model and generator
    model_interface = get_model_interface(model_config)
    generator = SummaryGenerator(model_interface, generation_config)
    
    # Process dataframe
    df = generator.process_dataframe(
        df, 
        prompt_column, 
        hierarchical_summary_column, 
        output_column
    )
    
    # Save results - only ID and generated summary
    output_df = df[['id', output_column]]
    output_df.to_csv(processing_config.output_path, index=False)
    logger.info(f"Generation complete. Results saved to {processing_config.output_path}")