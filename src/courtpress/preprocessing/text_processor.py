import pandas as pd
import numpy as np
import spacy
from typing import List, Union, Optional, Callable
from tqdm.auto import tqdm


class TextProcessor:
    """Text preprocessing utilities for court decisions and press releases."""

    def __init__(self, use_gpu: bool = False, batch_size: int = 32):
        """
        Initialize the text processor.

        Args:
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for processing (larger batches use more memory)
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.nlp = self._load_spacy_model()

    def _load_spacy_model(self) -> spacy.language.Language:
        """
        Load SpaCy language model with GPU acceleration if available and requested.

        Returns:
            Loaded SpaCy model
        """
        try:
            # Check if GPU acceleration for Spacy is available
            if self.use_gpu:
                # Try to load GPU-accelerated Spacy model
                try:
                    import spacy_cuda
                    nlp = spacy_cuda.load("de_core_news_sm")
                    print("SpaCy German model loaded with GPU acceleration")
                    return nlp
                except:
                    # Fallback to CPU version if GPU version fails
                    nlp = spacy.load("de_core_news_sm")
                    print("SpaCy German model loaded (CPU version)")
                    return nlp
            else:
                # Use regular Spacy on CPU
                nlp = spacy.load("de_core_news_sm")
                print("SpaCy German model loaded")
                return nlp
        except:
            print("SpaCy German model is being installed...")
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "de_core_news_sm"])
            nlp = spacy.load("de_core_news_sm")
            print("SpaCy German model loaded")
            return nlp

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        - Lemmatization
        - Removal of stopwords, punctuation, and short words
        - Keep only nouns, verbs, adjectives, and adverbs

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if pd.isna(text) or not text.strip():
            return "empty_document"  # Return a placeholder for empty texts

        doc = self.nlp(text)
        # Only keep nouns, verbs, adjectives, and adverbs
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct
                  and token.is_alpha and len(token.text) > 2
                  and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

        # If no tokens left after filtering, return at least some important words from the original text
        if not tokens:
            # Fallback: use any non-stopword, longer than 2 chars
            tokens = [token.lemma_.lower() for token in doc
                      if token.is_alpha and len(token.text) > 2 and not token.is_punct]

            # If still no tokens, use the first few words of the original text
            if not tokens and text:
                tokens = text.split()[:5]  # Take up to 5 words

            # If still empty, use a placeholder
            if not tokens:
                return "empty_document"

        return " ".join(tokens)

    def batch_preprocess_texts(self, texts: Union[List[str], pd.Series]) -> pd.Series:
        """
        Preprocess a batch of texts with optimized memory usage.

        Args:
            texts: List or pandas Series of texts to preprocess

        Returns:
            Pandas Series of preprocessed texts
        """
        if isinstance(texts, pd.Series):
            texts_list = texts.fillna("").tolist()
        else:
            texts_list = [text if pd.notna(text) else "" for text in texts]

        result = []

        for i in tqdm(range(0, len(texts_list), self.batch_size), desc="Preprocessing"):
            batch = texts_list[i:i+self.batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]
            result.extend(processed_batch)

            # Explicit memory cleanup for GPU
            if i % (self.batch_size * 10) == 0 and self.use_gpu:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except ImportError:
                    pass

        return pd.Series(result, index=range(len(result)))

    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a dataset by preprocessing the relevant text columns.

        Args:
            df: DataFrame with 'summary' and 'judgement' columns

        Returns:
            DataFrame with added preprocessed columns
        """
        df_copy = df.copy()
        print("Preprocessing press releases...")
        df_copy['preprocessed_summary'] = self.batch_preprocess_texts(
            df_copy['summary'])

        print("Preprocessing court decisions...")
        df_copy['preprocessed_judgement'] = self.batch_preprocess_texts(
            df_copy['judgement'])

        return df_copy
