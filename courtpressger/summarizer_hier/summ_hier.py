import os
import time
import argparse
import json
import math
import ast
from tqdm import tqdm
from collections import defaultdict

import pandas as pd

# Hugging Face imports
from transformers import pipeline, AutoTokenizer
import torch


class Summarizer():
    def __init__(
        self,
        chunk_size,
        max_context_len,
        max_summary_len,
        model_name,
        prompts,
        validate_summary=False,
        num_attempts=3,
        word_ratio=0.65,
        column_name=None        
    ):
        """
        :param chunk_size: Number of tokens per chunk at the initial level.
        :param max_context_len: Maximum tokens that can be processed at a time.
        :param max_summary_len: Maximum tokens in each final summary generation.
        :param word_ratio: Used for adjusting the word limit if a summary is invalid, etc.
        :param validate_summary: Whether to validate the summary for length & punctuation.
        :param num_attempts: How many times to attempt re-generation if invalid.
        :param model_name: Hugging Face model identifier or local model path.
        :param prompts: Path to the folder containing prompt templates.
        """

        print("=== Initializing Summarizer with the following arguments ===")
        print(f" chunk_size      : {chunk_size}")
        print(f" max_context_len : {max_context_len}")
        print(f" max_summary_len : {max_summary_len}")
        print(f" model_name      : {model_name}")
        print(f" prompts         : {prompts}")
        print(f" validate_summary: {validate_summary}")
        print(f" num_attempts    : {num_attempts}")
        print(f" word_ratio      : {word_ratio}")
        print(f" column_name     : {column_name}")
        print("============================================================\n")

        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.max_summary_len = max_summary_len
        self.word_ratio = word_ratio
        self.prompts = prompts
        self.model_name = model_name
        self.validate_summary = validate_summary
        self.num_attempts = num_attempts
        self.column_name = column_name

        # Load your local model & tokenizer
        print("Loading tokenizer and pipeline...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.client = pipeline(
            "text-generation",
            model=self.model_name,
            return_full_text=False,
            do_sample=False,
            repetition_penalty=1.2,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16
        )
        print("Tokenizer and pipeline loaded.\n")

        init_template_path = f"{self.prompts}/init.txt"
        merge_template_path = f"{self.prompts}/merge.txt"
        merge_context_path = f"{self.prompts}/merge_context.txt"

        print("Loading prompt templates:")
        print(f" init_template_path  : {init_template_path}")
        print(f" merge_template_path : {merge_template_path}")
        print(f" merge_context_path  : {merge_context_path}")
        self.templates = {
            'init_template': open(init_template_path, "r").read(),
            'template': open(merge_template_path, "r").read(),
            'context_template': open(merge_context_path, "r").read()
        }
        print("Templates loaded successfully.\n")

    def count_tokens(self, text):
        """Counts the number of tokens in a text using the tokenizer."""
        return len(self.tokenizer.encode(text))

    def obtain_response(self, prompt, max_tokens, temperature):
        """
        Calls the text-generation pipeline to obtain a response.
        We interpret 'max_tokens' as 'max_new_tokens' here.
        """
        print(f"[obtain_response] Generating text with max_tokens={max_tokens}")
        try:
            outputs = self.client(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            response = outputs[0]['generated_text'].strip()
            print(f"[obtain_response] Response length (chars): {len(response)}\n")
            return response
        except Exception as e:
            print(f"[obtain_response] Generation error: {e}")
            return ""

    def check_summary_validity(self, summary, token_limit):
        """Heuristic checks for an acceptable summary length and ending punctuation."""
        print("[check_summary_validity] Checking if summary is valid...")
        if len(summary) == 0:
            print("[check_summary_validity] Summary is empty.")
            return False
        # If the summary is too long or doesn't end with punctuation, consider it invalid
        if self.count_tokens(summary) > token_limit or summary[-1] not in ['.', '?', '!', '\"', '\'']:
            print("[check_summary_validity] Summary is invalid due to length or ending punctuation.\n")
            return False
        print("[check_summary_validity] Summary is valid.\n")
        return True

    def summarize_texts(self, texts, token_limit, level):
        """
        Summarize a given text (plus optional context) based on a given token_limit.
        """
        text = texts['text']
        context = texts['context']
        print(f"[summarize_texts] Summarizing at level {level} with token_limit={token_limit}.")

        word_limit = round(token_limit * self.word_ratio)

        # Create the prompt
        if level == 0:
            prompt = self.templates['init_template'].format(text, word_limit)
        else:
            prompt = self.templates['template'].format(text, word_limit)
            if len(context) > 0 and level > 0:
                prompt = self.templates['context_template'].format(context, text, word_limit)

        response = self.obtain_response(prompt, max_tokens=token_limit, temperature=None)

        # Retry if empty
        while len(response) == 0:
            print("[summarize_texts] Received an empty summary, retrying in 10 seconds...")
            time.sleep(10)
            response = self.obtain_response(prompt, max_tokens=token_limit, temperature=None)

        # Validate summary (if enabled)
        attempts = 0
        while not self.check_summary_validity(response, token_limit) and self.validate_summary:
            attempts += 1
            if attempts > self.num_attempts:
                print(f"[summarize_texts] Failed to generate valid summary after {self.num_attempts} attempts. Skipping.\n")
                return response

            # Adjust word limit downward
            word_limit = int(word_limit * (1 - 0.1 * attempts))
            print(f"[summarize_texts] Invalid summary. Retrying with a reduced word_limit={word_limit}. Attempt {attempts}/{self.num_attempts}.")

            if level == 0:
                prompt = self.templates['init_template'].format(text, word_limit)
            else:
                prompt = self.templates['template'].format(text, word_limit)
                if len(context) > 0 and level > 0:
                    prompt = self.templates['context_template'].format(context, text, word_limit)

            response = self.obtain_response(prompt, max_tokens=token_limit, temperature=None)

        print(f"[summarize_texts] Final summary length (tokens): {self.count_tokens(response)}\n")
        return response

    def estimate_levels(self, ruling_chunks, summary_limit=450):
        """
        Estimate how many hierarchical levels are needed given the chunking
        and build an appropriate list of summary token-limits.
        """
        num_chunks = len(ruling_chunks)
        chunk_limit = self.chunk_size
        levels = 0

        print(f"[estimate_levels] Estimating levels for {num_chunks} chunks...")

        # Repeatedly merge chunks until only one remains => define how many levels are needed
        while num_chunks > 1:
            # number of chunks that could fit into the current context at once
            chunks_that_fit = (
                self.max_context_len
                - self.count_tokens(self.templates['template'].format('', 0))
                - 20
            ) // chunk_limit

            num_chunks = math.ceil(num_chunks / max(1, chunks_that_fit))
            chunk_limit = summary_limit
            levels += 1

        # Build the list of allowable summary lengths from bottom to top
        summary_limits = [self.max_summary_len]
        for _ in range(levels - 1):
            summary_limits.append(int(summary_limits[-1] * self.word_ratio))
        summary_limits.reverse()

        print(f"[estimate_levels] Total levels: {levels}")
        print(f"[estimate_levels] Summary limits by level (bottom->top): {summary_limits}\n")

        return levels, summary_limits

    def recursive_summary(self, summaries, level, chunks, summary_limits):
        """
        Merges chunks into summaries recursively until the result is small enough
        to be summarized in one go.
        """
        print(f"[recursive_summary] Merging {len(chunks)} chunks at level {level}...")

        i = 0
        summaries_dict = summaries['summaries_dict']

        if level >= len(summary_limits):
            summary_limit = self.max_summary_len
        else:
            summary_limit = summary_limits[level]

        # Check if we can fit everything in one shot at this level
        if level > 0 and len(summaries_dict[level]) > 0:
            # If context + chunk content + summary template fits, use max_summary_len
            if (
                self.count_tokens('\n\n'.join(chunks))
                + self.max_summary_len
                + self.count_tokens(self.templates['context_template'].format('', '', 0))
                + 20
                <= self.max_context_len
            ):
                summary_limit = self.max_summary_len

            num_tokens = (
                self.max_context_len
                - summary_limit
                - self.count_tokens(self.templates['context_template'].format('', '', 0))
                - 20
            )
        else:
            if (
                self.count_tokens('\n\n'.join(chunks))
                + self.max_summary_len
                + self.count_tokens(self.templates['template'].format('', 0))
                + 20
                <= self.max_context_len
            ):
                summary_limit = self.max_summary_len

            num_tokens = (
                self.max_context_len
                - summary_limit
                - self.count_tokens(self.templates['template'].format('', 0))
                - 20
            )

        while i < len(chunks):
            # Generate context from the last summary at the current level
            context = summaries_dict[level][-1] if len(summaries_dict[level]) > 0 else ""
            context_len = math.floor(0.2 * num_tokens)

            # If context is too large, trim it
            if self.count_tokens(context) > context_len:
                context_tokens = self.tokenizer.encode(context)[:context_len]
                context = self.tokenizer.decode(context_tokens)
                if '.' in context:
                    context = context.rsplit('.', 1)[0] + '.'

            # Concatenate as many chunks as we can fit
            if level == 0:
                text = chunks[i]
            else:
                j = 1
                text = f"Summary {j}:\n\n{chunks[i]}"
                while (
                    i + 1 < len(chunks)
                    and self.count_tokens(context + text + f"\n\nSummary {j+1}:\n\n{chunks[i+1]}") + 20 <= num_tokens
                ):
                    i += 1
                    j += 1
                    text += f"\n\nSummary {j}:\n\n{chunks[i]}"

            texts = {'text': text, 'context': context}
            summary = self.summarize_texts(texts, summary_limit, level)
            summaries_dict[level].append(summary)
            i += 1

        # If there's more than one chunk of summaries, recursively merge
        if len(summaries_dict[level]) > 1:
            print(f"[recursive_summary] Level {level} produced {len(summaries_dict[level])} summaries; proceeding to next level.\n")
            return self.recursive_summary(summaries, level + 1, summaries_dict[level], summary_limits)
        else:
            # final summary
            print(f"[recursive_summary] Level {level} final summary obtained.\n")
            return summaries_dict[level][0]

    def summarize_ruling(self, chunks):
        """
        Summarize a single ruling's chunks hierarchically.
        """
        print(f"[summarize_ruling] Summarizing ruling with {len(chunks)} chunks...\n")
        # Decide how many levels we might need
        levels, summary_limits = self.estimate_levels(chunks)
        level = 0

        # Create a fresh dictionary for intermediate summaries
        summaries = {
            'summaries_dict': defaultdict(list)
        }

        final_summary = self.recursive_summary(summaries, level, chunks, summary_limits)
        return final_summary

    def get_summaries_for_dataframe(self, df):
        """
        Given a DataFrame (with a 'chunks' column),
        produce hierarchical summaries for each row and store them in 'final_summary'.
        """
        print("[get_summaries_for_dataframe] Generating summaries for the DataFrame...\n")
        final_summaries = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing rulings"):
            ruling_chunks = row["chunks"]
            final_summary = self.summarize_ruling(ruling_chunks)
            final_summaries.append(final_summary)

        column_name = self.column_name
        df[column_name] = final_summaries
        print("[get_summaries_for_dataframe] Summaries generation completed.\n")
        return df


def summarize_data(args):
    """
    Summarizes the data from a CSV where each row has a 'chunks' column
    (stringified list of text chunks). Writes the summarized results back to a CSV.
    """
    print("=== summarize_data called with the following arguments ===")
    for arg in vars(args):
        print(f" {arg}: {getattr(args, arg)}")
    print("==========================================================\n")

    # Load the data
    print(f"Loading data from {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Data loaded. Rows: {len(df)}\n")

    # Convert stringified list in 'chunks' column to Python list
    print("Converting stringified chunks to Python lists...")
    df["chunks"] = df["chunks"].apply(lambda x: ast.literal_eval(x))
    print("Conversion complete.\n")

    # Initialize Summarizer
    summarizer = Summarizer(
        chunk_size=args.chunk_size,
        max_context_len=args.context_len,
        max_summary_len=args.summary_len,
        model_name=args.model,
        prompts=args.prompts,
        validate_summary=args.validate_summary,
        num_attempts=args.num_attempts,
        word_ratio=args.word_ratio,
        column_name=args.column_name
    )

    # Summarize the data
    print("Starting the summarization process for all rows...\n")
    df_summarized = summarizer.get_summaries_for_dataframe(df)

    # Save to CSV
    print(f"Saving summarized data to {args.output} ...")
    df_summarized.to_csv(args.output, index=False)
    print("Data saved successfully.\n")