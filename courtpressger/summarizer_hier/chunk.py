# summarizer_hier/chunk.py

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import json
from .model_interface import get_model_interface

class TextChunker:
    """
    A class to handle token counting and text chunking for large documents.
    """
    def __init__(self, model_interface, chunk_size=1024):
        """
        Initialize the TextChunker with a model interface and a maximum chunk size.
        
        :param model_interface: An object that implements `ModelInterface`.
        :param chunk_size: The maximum number of tokens allowed per chunk.
        """
        self.model_interface = model_interface
        self.chunk_size = chunk_size

    def count_tokens(self, text):
        """
        Return the number of tokens in 'text' using the given model interface.
        """
        return self.model_interface.count_tokens(text)

    def find_punctuations(self, text, comma=False):
        """
        Find indices of punctuation marks in the text. 
        If comma=True, also consider commas as punctuation.
        """
        if comma:
            puncs = ['.', '?', '!', ',', '."', '?"', '!"', ".'", "?'", "!'"]
        else:
            puncs = ['.', '?', '!', '."', '?"', '!"', ".'", "?'", "!'"]
        
        puncs_idx = []
        
        for i, c in enumerate(text):
            # Direct single-character punctuation check
            if c in puncs:
                puncs_idx.append(i)
            # Handle quotes that follow ., ?, or ! (e.g. '."' or '!"')
            elif c in ['"', "'"] and i > 0:
                if text[i-1] in ['.', '?', '!']:
                    puncs_idx.append(i)
        return puncs_idx

    def truncate_text(self, text):
        """
        Truncate 'text' at the last punctuation so it does not exceed 'self.chunk_size' tokens.
        Returns (truncated_text, remainder).
        """
        original_text = text
        # Keep truncating until token count is within limit
        while self.count_tokens(text) > self.chunk_size:
            puncs_idx = self.find_punctuations(text)
            try:
                # Attempt to truncate at the second-to-last punctuation
                text = text[: puncs_idx[-2] + 1]
            except IndexError:
                # If not enough punctuation found, try again with commas
                puncs_idx = self.find_punctuations(text, comma=True)
                try:
                    text = text[: puncs_idx[-2] + 1]
                except:
                    # If we can't safely truncate, return what's possible
                    return text, ''
        truncated = original_text[len(text):]
        return text, truncated

    def chunk_text(self, paragraphs):
        """
        Given a list of paragraphs, combine them into chunks without exceeding 'self.chunk_size' tokens.
        """
        chunks = []
        current_chunk = ''

        for paragraph in paragraphs:
            new_chunk = '\n'.join([current_chunk, paragraph]) if current_chunk else paragraph

            # If a single paragraph is too long, we truncate it
            if self.count_tokens(paragraph) > self.chunk_size:
                truncated_text, remainder = self.truncate_text(new_chunk)
                chunks.append(truncated_text)

                # Continue truncating the remainder until it fits
                while remainder and self.count_tokens(remainder) > self.chunk_size:
                    truncated_text, remainder = self.truncate_text(remainder)
                    chunks.append(truncated_text)
                current_chunk = remainder
                continue
            
            # If adding this paragraph exceeds the limit, finalize the current chunk
            if self.count_tokens(new_chunk) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk = new_chunk

        # Append any leftover chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_judgement(self, text):
        """
        Takes the full text of a single 'judgement' and returns a list of its chunks.
        """
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p for p in paragraphs if p.strip()]

        combined_paragraphs = '\n'.join(paragraphs)
        # If the entire text fits within the limit, return as one chunk
        if self.count_tokens(combined_paragraphs) <= self.chunk_size:
            return [combined_paragraphs]
        else:
            chunks = self.chunk_text(paragraphs)
            
            # (Optional) verify no token loss
            original_tokens = self.count_tokens(''.join(paragraphs).replace('\n', ''))
            chunked_tokens = self.count_tokens(''.join(chunks).replace('\n', ''))
            diff = original_tokens - chunked_tokens
            if diff != 0:
                print(f"Warning: Token difference is {diff}")

            return chunks


def chunk_data(config):
    """
    Main function to read the input CSV, chunk the 'judgement' column, and save the results.
    """
    print("\n=== [chunk_data] Using config ===")
    print(json.dumps(config, indent=2))
    print("================================\n")

    # Proceed with data loading
    df = pd.read_csv(config["input"])
    if 'judgement' not in df.columns or 'synthetic_prompt' not in df.columns:
        raise ValueError("DataFrame must contain 'judgement' and 'synthetic_prompt' columns.")

    # Build the model interface from config
    model_interface = get_model_interface(config)

    # Initialize the chunker
    chunker = TextChunker(model_interface, config["chunk_size"])

    all_chunks = []
    all_chunk_token_counts = []

    for text in tqdm(df["judgement"], desc="Chunking Judgements", total=len(df)):
        chunks = chunker.chunk_judgement(text)
        all_chunks.append(chunks)
        for ch in chunks:
            all_chunk_token_counts.append(chunker.count_tokens(ch))

    df["chunks"] = all_chunks
    df = df[["id","split_name", "chunks"]]
    df.to_csv(config["output"], index=False)
    print(df)

    # Print stats
    if all_chunk_token_counts:
        p25, p50, p75, p99 = np.percentile(all_chunk_token_counts, [25, 50, 75, 99])
        maximum = max(all_chunk_token_counts)
        mean = np.mean(all_chunk_token_counts)
        std_dev = np.std(all_chunk_token_counts)
        count = len(all_chunk_token_counts)
        stats_str = (
            "\n=== Chunk Token Count Statistics ===\n"
            f"Count of all chunks:      {count}\n"
            f"25th Percentile:          {p25:.2f}\n"
            f"50th Percentile (median): {p50:.2f}\n"
            f"75th Percentile:          {p75:.2f}\n"
            f"99th Percentile:          {p99:.2f}\n"
            f"Max:                      {maximum}\n"
            f"Mean:                     {mean:.2f}\n"
            f"Std. Dev:                 {std_dev:.2f}\n"
            "======================================\n"
        )
        print(stats_str)
    else:
        print("No chunks were produced, so no statistics can be shown.")