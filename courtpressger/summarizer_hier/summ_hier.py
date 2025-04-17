# summarizer_hier/summ_hier.py

import os
import time
import argparse
import json
import math
import ast
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
from .model_interface import get_model_interface

class Summarizer():
    def __init__(
        self,
        chunk_size,
        max_context_len,
        max_summary_len,
        model_interface,
        prompts,
        validate_summary=False,
        num_attempts=3,
        word_ratio=0.65,
        column_name=None        
    ):
        """
        :param model_interface: An object implementing `ModelInterface` 
                                (for token counting + text generation).
        :param chunk_size: Number of tokens per chunk at the initial level.
        :param max_context_len: Maximum tokens that can be processed at a time.
        :param max_summary_len: Maximum tokens in each final summary generation.
        :param prompts: Path to folder containing prompt templates.
        :param validate_summary: Whether to validate summary for length, punctuation.
        :param num_attempts: How many times to attempt re-generation if invalid.
        :param word_ratio: For adjusting word limit if summary is invalid, etc.
        :param column_name: Column name for the final summary output.
        """

        print("=== Initializing Summarizer with the following arguments ===")
        print(f" chunk_size      : {chunk_size}")
        print(f" max_context_len : {max_context_len}")
        print(f" max_summary_len : {max_summary_len}")
        print(f" prompts         : {prompts}")
        print(f" validate_summary: {validate_summary}")
        print(f" num_attempts    : {num_attempts}")
        print(f" word_ratio      : {word_ratio}")
        print(f" column_name     : {column_name}")
        print("============================================================\n")

        self.model_interface = model_interface
        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.max_summary_len = max_summary_len
        self.word_ratio = word_ratio
        self.prompts = prompts
        self.validate_summary = validate_summary
        self.num_attempts = num_attempts
        self.column_name = column_name

        model_name = getattr(model_interface, "model_name", "").lower()
        self.is_teuken = "teuken" in model_name
        self.system_message_de = (
            "Ein Gespräch zwischen einem Menschen und einem Assistenten "
            "mit künstlicher Intelligenz. Der Assistent gibt hilfreiche "
            "und höfliche Antworten auf die Fragen des Menschen."
        )

        # Load prompt templates
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

    def _build_teuken_prompt(self, prompt: str) -> str:
        """
        Build a chat-style Teuken prompt of the form:
            System: ...
            User: ...
            Assistant:
        """
        prompt_chat = (
            f"System: {self.system_message_de}\n"
            f"User: {prompt}\n"
            "Assistant:"
        )
        return prompt_chat

    def count_tokens(self, text):
        """
        Counts the number of tokens in a text using the model interface.
        """
        return self.model_interface.count_tokens(text)

    def obtain_response(self, prompt, max_tokens, temperature):
        """
        Calls the model interface to obtain a response.
        """
        return self.model_interface.generate_text(prompt, max_tokens, temperature)

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
        Summarize the given text (plus optional context) based on a given token_limit.
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
        

        prompt = [{'role': 'user', 'content': prompt}]

        if self.is_teuken:
            prompt = self.model_interface.tokenizer.apply_chat_template(prompt, tokenize=False, chat_template="DE",add_generation_prompt=True)

        response = self.obtain_response(prompt, max_tokens=token_limit, temperature=0.1)

        # Retry if empty
        while len(response) == 0:
            print("[summarize_texts] Received an empty summary, retrying in 10 seconds...")
            time.sleep(10)
            response = self.obtain_response(prompt, max_tokens=token_limit, temperature=0.1)

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

            response = self.obtain_response(prompt, max_tokens=token_limit, temperature=0.1)

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

        # Repeatedly merge chunks until only one remains => define how many levels
        while num_chunks > 1:
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

        # If there's context from previous chunk at the same level, we do some fitting logic
        if level > 0 and len(summaries_dict[level]) > 0:
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
                context_tokens = self.model_interface.tokenizer.encode(context)[:context_len]
                context = self.model_interface.tokenizer.decode(context_tokens)
                if '.' in context:
                    context = context.rsplit('.', 1)[0] + '.'

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

        # If there's more than one chunk, we must merge them at the next level
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
        # Decide how many levels
        levels, summary_limits = self.estimate_levels(chunks)
        level = 0

        summaries = {'summaries_dict': defaultdict(list)}
        final_summary = self.recursive_summary(summaries, level, chunks, summary_limits)
        return final_summary

    def get_summaries_for_dataframe(self, df):
        """
        Given a DataFrame (with 'chunks' column) produce hierarchical summaries 
        for each row and store them in 'final_summary' or the chosen column name.
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


def summarize_data(config):
    """
    Summarizes the data from a CSV. 
    Expects 'chunks' column in the CSV with stringified lists.
    """
    print("=== summarize_data ===")
    print("Config received:", config, "\n")

    # Load the data
    df = pd.read_csv(config["input"])
    print(f"Data loaded. Rows: {len(df)}\n")

    # Convert stringified 'chunks' into lists
    df["chunks"] = df["chunks"].apply(lambda x: ast.literal_eval(x))

    # Instantiate the model interface
    model_interface = get_model_interface(config)

    # Create Summarizer
    summarizer = Summarizer(
        chunk_size=config["chunk_size"],
        max_context_len=config["context_len"],
        max_summary_len=config["summary_len"],
        model_interface=model_interface,
        prompts=config["prompts"],
        validate_summary=config.get("validate_summary", False),
        num_attempts=config.get("num_attempts", 3),
        word_ratio=config.get("word_ratio", 0.65),
        column_name=config["column_name"]
    )

    # Summarize data
    df_summarized = summarizer.get_summaries_for_dataframe(df)

    # Save results
    df_summarized.to_csv(config["output"], index=False)
    print(f"Data saved to {config['output']}.\n")
