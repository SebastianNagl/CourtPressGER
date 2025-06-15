"""Text chunking module for splitting large documents."""

import re
import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

from .config import ConfigManager, ChunkConfig
from .model_interface import get_model_interface

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles splitting of large texts into manageable chunks."""
    
    def __init__(self, model_interface, chunk_config: ChunkConfig):
        """
        Initialize the TextChunker.
        
        Args:
            model_interface: Model interface for token counting
            chunk_config: Chunking configuration
        """
        self.model_interface = model_interface
        self.chunk_size = chunk_config.chunk_size
        self.overlap = chunk_config.overlap
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model interface."""
        return self.model_interface.count_tokens(text)
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """Find indices of sentence boundaries in text."""
        # Sentence-ending punctuation patterns
        patterns = [
            r'\. ',  # Period followed by space
            r'\.\n',  # Period followed by newline
            r'\? ',  # Question mark followed by space
            r'\?\n',  # Question mark followed by newline
            r'! ',   # Exclamation followed by space
            r'!\n',  # Exclamation followed by newline
            r'\." ',  # Period and quote followed by space
            r'\?" ',  # Question and quote followed by space
            r'!" ',   # Exclamation and quote followed by space
        ]
        
        boundaries = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.end() - 1)
        
        return sorted(set(boundaries))
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks respecting token limits and sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # First, split into paragraphs
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If the entire text fits in one chunk, return it
        if self.count_tokens(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If a single paragraph is too large, split it
            if paragraph_tokens > self.chunk_size:
                # Split paragraph at sentence boundaries
                sentences = self._split_paragraph_into_sentences(paragraph)
                
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens += sentence_tokens
            
            # Check if adding this paragraph exceeds limit
            elif current_tokens + paragraph_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into sentences."""
        # Simple sentence splitting - can be improved with better NLP tools
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Main method to chunk a document.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = self.split_text(text)
        
        # Log statistics
        token_counts = [self.count_tokens(chunk) for chunk in chunks]
        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"Token counts - Min: {min(token_counts)}, Max: {max(token_counts)}, "
                   f"Avg: {np.mean(token_counts):.1f}")
        
        return chunks


def run_chunking(config_manager: ConfigManager, overrides: dict) -> None:
    """
    Run the chunking process.
    
    Args:
        config_manager: Configuration manager
        overrides: Configuration overrides
    """
    # Get configurations
    model_config = config_manager.get_model_config(overrides.get('model'))
    chunk_config = config_manager.get_chunk_config(overrides.get('chunking'))
    processing_config = config_manager.get_processing_config(overrides.get('processing'))
    
    logger.info("Starting chunking process...")
    logger.info(f"Input: {processing_config.input_path}")
    logger.info(f"Output: {processing_config.output_path}")
    logger.info(f"Chunk size: {chunk_config.chunk_size}")
    
    # Load data
    df = pd.read_csv(processing_config.input_path)
    judgement_column = processing_config.column_mappings['judgement_column']
    
    if judgement_column not in df.columns:
        raise ValueError(f"Column '{judgement_column}' not found in input data")
    
    # Initialize model interface and chunker
    model_interface = get_model_interface(model_config)
    chunker = TextChunker(model_interface, chunk_config)
    
    # Process documents
    all_chunks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        text = row[judgement_column]
        chunks = chunker.chunk_document(text)
        all_chunks.append(chunks)
    
    # Add chunks to dataframe
    chunks_column = processing_config.column_mappings['chunks_column']
    df[chunks_column] = all_chunks
    
    # Save results
    df.to_csv(processing_config.output_path, index=False)
    logger.info(f"Chunking complete. Results saved to {processing_config.output_path}")
    
    # Print overall statistics
    all_token_counts = []
    for chunks in all_chunks:
        all_token_counts.extend([chunker.count_tokens(chunk) for chunk in chunks])
    
    if all_token_counts:
        logger.info("\n=== Overall Chunk Statistics ===")
        logger.info(f"Total chunks: {len(all_token_counts)}")
        logger.info(f"Min tokens: {min(all_token_counts)}")
        logger.info(f"Max tokens: {max(all_token_counts)}")
        logger.info(f"Avg tokens: {np.mean(all_token_counts):.1f}")
        logger.info(f"Median tokens: {np.median(all_token_counts):.1f}")