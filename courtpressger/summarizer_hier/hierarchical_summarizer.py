"""Hierarchical summarization module."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import json

from .config import ConfigManager, SummarizationConfig
from .model_interface import get_model_interface

logger = logging.getLogger(__name__)


class HierarchicalSummarizer:
    """Performs hierarchical summarization of chunked documents."""
    
    def __init__(self, model_interface, config: SummarizationConfig):
        """
        Initialize the hierarchical summarizer.
        
        Args:
            model_interface: Model interface for text generation
            config: Summarization configuration
        """
        self.model_interface = model_interface
        self.config = config
        self.templates = self._load_templates()
        
        # Track intermediate results
        self.intermediate_results = []
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        templates = {}
        prompts_dir = Path(self.config.prompts_dir)
        
        template_files = {
            'initial': 'init.txt',
            'merge': 'merge.txt',
            'context': 'merge_context.txt'
        }
        
        for key, filename in template_files.items():
            filepath = prompts_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    templates[key] = f.read()
                logger.debug(f"Loaded template: {key}")
            else:
                logger.warning(f"Template file not found: {filepath}")
                # Provide default templates
                if key == 'initial':
                    templates[key] = "Summarize the following text in approximately {0} words:\n\n{1}"
                elif key == 'merge':
                    templates[key] = "Merge and summarize the following summaries in approximately {0} words:\n\n{1}"
                elif key == 'context':
                    templates[key] = "Given the context:\n{0}\n\nSummarize the following in approximately {1} words:\n\n{2}"
        
        return templates
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.model_interface.count_tokens(text)
    
    def generate_summary(self, prompt: str, max_tokens: int, temperature: float = 0.1) -> str:
        """Generate a summary using the model."""
        messages = [{'role': 'user', 'content': prompt}]
        return self.model_interface.generate_text(messages, max_tokens, temperature)
    
    def validate_summary(self, summary: str, token_limit: int) -> bool:
        """Validate that a summary meets quality criteria."""
        if not summary:
            return False
        
        # Check length
        if self.count_tokens(summary) > token_limit:
            return False
        
        # Check ending punctuation
        if not summary.rstrip().endswith(('.', '!', '?', '"', "'")):
            return False
        
        return True
    
    def summarize_chunk(self, 
                       text: str, 
                       context: Optional[str] = None,
                       level: int = 0,
                       max_tokens: int = None) -> str:
        """
        Summarize a single chunk of text.
        
        Args:
            text: Text to summarize
            context: Optional context from previous summaries
            level: Hierarchical level (0 = initial chunks)
            max_tokens: Maximum tokens for summary
            
        Returns:
            Generated summary
        """
        if max_tokens is None:
            max_tokens = self.config.max_summary_length
        
        # Calculate word limit
        word_limit = int(max_tokens * self.config.word_ratio)
        
        # Select appropriate template
        if level == 0:
            prompt = self.templates['initial'].format(word_limit, text)
        elif context:
            prompt = self.templates['context'].format(context, word_limit, text)
        else:
            prompt = self.templates['merge'].format(word_limit, text)
        
        # Generate summary with retries
        attempts = 0
        while attempts < self.config.num_attempts:
            summary = self.generate_summary(prompt, max_tokens, self.config.temperature)
            
            if self.config.validate_summary:
                if self.validate_summary(summary, max_tokens):
                    return summary
                else:
                    attempts += 1
                    # Reduce word limit for next attempt
                    word_limit = int(word_limit * 0.9)
                    logger.debug(f"Retrying with reduced word limit: {word_limit}")
            else:
                return summary
        
        # Return whatever we got after max attempts
        return summary
    
    def merge_summaries(self, 
                       summaries: List[str], 
                       level: int,
                       max_tokens: int = None) -> List[str]:
        """
        Merge multiple summaries into fewer summaries.
        
        Args:
            summaries: List of summaries to merge
            level: Current hierarchical level
            max_tokens: Maximum tokens per merged summary
            
        Returns:
            List of merged summaries
        """
        if max_tokens is None:
            max_tokens = self.config.max_summary_length
        
        merged = []
        current_batch = []
        current_tokens = 0
        
        # Calculate space needed for template
        template_tokens = self.count_tokens(self.templates['merge'].format('', ''))
        available_tokens = self.config.max_context_length - max_tokens - template_tokens - 50
        
        for i, summary in enumerate(summaries):
            summary_text = f"Summary {len(current_batch) + 1}:\n\n{summary}"
            summary_tokens = self.count_tokens(summary_text)
            
            if current_tokens + summary_tokens > available_tokens and current_batch:
                # Merge current batch
                merged_text = "\n\n".join(current_batch)
                merged_summary = self.summarize_chunk(merged_text, level=level, max_tokens=max_tokens)
                merged.append(merged_summary)
                
                # Start new batch
                current_batch = [summary_text]
                current_tokens = summary_tokens
            else:
                current_batch.append(summary_text)
                current_tokens += summary_tokens
        
        # Process remaining batch
        if current_batch:
            merged_text = "\n\n".join(current_batch)
            merged_summary = self.summarize_chunk(merged_text, level=level, max_tokens=max_tokens)
            merged.append(merged_summary)
        
        return merged
    
    def calculate_levels(self, num_chunks: int) -> Tuple[int, List[int]]:
        """
        Calculate the number of hierarchical levels needed.
        
        Args:
            num_chunks: Number of initial chunks
            
        Returns:
            Tuple of (number of levels, summary length limits per level)
        """
        levels = 0
        current_chunks = num_chunks
        
        while current_chunks > 1:
            # Estimate how many chunks can be merged at once
            chunks_per_merge = max(2, (self.config.max_context_length - self.config.max_summary_length) // self.config.chunk_size)
            current_chunks = (current_chunks + chunks_per_merge - 1) // chunks_per_merge
            levels += 1
        
        # Calculate summary lengths for each level
        summary_lengths = []
        current_length = self.config.max_summary_length
        
        for _ in range(levels):
            summary_lengths.append(current_length)
            current_length = int(current_length * self.config.word_ratio)
        
        summary_lengths.reverse()  # Bottom to top
        
        return levels, summary_lengths
    
    def summarize_hierarchically(self, 
                                chunks: List[str],
                                document_id: str = None) -> str:
        """
        Perform hierarchical summarization on a list of chunks.
        
        Args:
            chunks: List of text chunks
            document_id: Optional document identifier for tracking
            
        Returns:
            Final summary
        """
        if not chunks:
            return ""
        
        # Single chunk - just summarize it
        if len(chunks) == 1:
            return self.summarize_chunk(chunks[0], level=0)
        
        # Calculate levels needed
        num_levels, summary_lengths = self.calculate_levels(len(chunks))
        logger.info(f"Document {document_id}: {len(chunks)} chunks, {num_levels} levels")
        
        # Level 0: Summarize individual chunks
        current_summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc=f"Level 0", leave=False)):
            summary = self.summarize_chunk(
                chunk, 
                level=0, 
                max_tokens=summary_lengths[0] if summary_lengths else self.config.max_summary_length
            )
            current_summaries.append(summary)
            
            # Track intermediate result
            if document_id:
                self.intermediate_results.append({
                    'document_id': document_id,
                    'level': 0,
                    'chunk_index': i,
                    'summary': summary
                })
        
        # Higher levels: Merge summaries
        for level in range(1, num_levels + 1):
            if len(current_summaries) == 1:
                break
            
            max_tokens = summary_lengths[level] if level < len(summary_lengths) else self.config.max_summary_length
            current_summaries = self.merge_summaries(current_summaries, level, max_tokens)
            
            # Track intermediate results
            if document_id:
                for i, summary in enumerate(current_summaries):
                    self.intermediate_results.append({
                        'document_id': document_id,
                        'level': level,
                        'chunk_index': i,
                        'summary': summary
                    })
            
            logger.info(f"Level {level}: {len(current_summaries)} summaries")
        
        # Return final summary
        return current_summaries[0] if current_summaries else ""
    
    def process_dataframe(self, df: pd.DataFrame, chunks_column: str, output_column: str) -> pd.DataFrame:
        """
        Process a dataframe with chunked documents.
        
        Args:
            df: Input dataframe
            chunks_column: Column containing chunks
            output_column: Column to store summaries
            
        Returns:
            Dataframe with summaries added
        """
        summaries = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing documents"):
            chunks = row[chunks_column]
            
            # Handle string representation of lists
            if isinstance(chunks, str):
                chunks = ast.literal_eval(chunks)
            
            document_id = row.get('id', idx)
            summary = self.summarize_hierarchically(chunks, document_id)
            summaries.append(summary)
        
        df[output_column] = summaries
        return df
    
    def save_intermediate_results(self, filepath: str) -> None:
        """Save intermediate summarization results."""
        if self.intermediate_results:
            df = pd.DataFrame(self.intermediate_results)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(self.intermediate_results)} intermediate results to {filepath}")


def run_summarization(config_manager: ConfigManager, overrides: dict) -> None:
    """
    Run the hierarchical summarization process.
    
    Args:
        config_manager: Configuration manager
        overrides: Configuration overrides
    """
    # Get configurations
    model_config = config_manager.get_model_config(overrides.get('model'))
    summarization_config = config_manager.get_summarization_config(overrides.get('summarization'))
    processing_config = config_manager.get_processing_config(overrides.get('processing'))
    
    logger.info("Starting hierarchical summarization...")
    logger.info(f"Input: {processing_config.input_path}")
    logger.info(f"Output: {processing_config.output_path}")
    
    # Load data
    df = pd.read_csv(processing_config.input_path)
    chunks_column = processing_config.column_mappings['chunks_column']
    
    if chunks_column not in df.columns:
        raise ValueError(f"Column '{chunks_column}' not found in input data")
    
    # Initialize model and summarizer
    model_interface = get_model_interface(model_config)
    summarizer = HierarchicalSummarizer(model_interface, summarization_config)
    
    # Process dataframe
    output_column = processing_config.column_mappings['hierarchical_summary_column']
    df = summarizer.process_dataframe(df, chunks_column, output_column)
    
    # Save results
    df.to_csv(processing_config.output_path, index=False)
    logger.info(f"Summarization complete. Results saved to {processing_config.output_path}")
    
    # Save intermediate results if requested
    intermediate_path = Path(processing_config.output_path).parent / "intermediate_summaries.csv"
    summarizer.save_intermediate_results(str(intermediate_path))