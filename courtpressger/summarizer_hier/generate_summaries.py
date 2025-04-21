import pandas as pd
from tqdm import tqdm
from .model_interface import get_model_interface


tqdm.pandas()

class Generator:
    def __init__(self, 
                max_context_len,
                max_summary_len,
                model_interface,
                prompt_column_name,
                hier_summ_column_name,
                output_column_name
    ):
        print("=== Initializing Generator with the following arguments ===")
        print(f" max_context_len : {max_context_len}")
        print(f" max_summary_len : {max_summary_len}")
        print(f" model_interface : {model_interface}")
        print(f" prompt_column_name : {prompt_column_name}")
        print(f" hier_summ_column_name : {hier_summ_column_name}")
        print(f" output_column_name : {output_column_name}")
        print("============================================================\n")

        self.max_context_len = max_context_len
        self.max_summary_len = max_summary_len
        self.model_interface = model_interface
        self.prompt_column_name = prompt_column_name
        self.hier_summ_column_name = hier_summ_column_name
        self.output_column_name = output_column_name

        model_name = getattr(model_interface, "model_name", "").lower()
        self.is_teuken = "teuken" in model_name
        self.system_message_de = (
            "Ein Gespräch zwischen einem Menschen und einem Assistenten "
            "mit künstlicher Intelligenz. Der Assistent gibt hilfreiche "
            "und höfliche Antworten auf die Fragen des Menschen."
        )

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

    def generate_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in the DataFrame, build a synthetic prompt by concatenating
        the prompt column and hierarchical summary column, then generate a summary.
        Returns a new DataFrame with only 'id' and the generated summary column.
        """
        # Use tqdm to track progress
        summaries = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
            # Construct the synthetic prompt
            synthetic_prompt = row[self.prompt_column_name]
            hier_summ = row[self.hier_summ_column_name]
            prompt = f"{synthetic_prompt}\n\n{hier_summ}"
            prompt = [{'role': 'user', 'content': prompt}]
            # Generate the summary
            summary = self.obtain_response(
                prompt,
                max_tokens=self.max_summary_len,
                temperature=0.1
            )
            summaries.append(summary)

        # Build output DataFrame with only 'id' and the summary column
        output_df = pd.DataFrame({
            'id': df['id'],
            self.output_column_name: summaries
        })
        return output_df


def generate_summaries(config: dict):
    """
    Summarizes the data from a CSV. 
    Expects 'chunks' column in the CSV with stringified lists.
    """
    print("=== summarize_data ===")
    print("Config received:", config, "\n")

    # Load the data
    df = pd.read_csv(config["input"])
    print(f"Data loaded. Rows: {len(df)}\n")

    # Instantiate the model interface
    model_interface = get_model_interface(config)

    # Create Generator
    generator = Generator(
        max_context_len=config["context_len"],
        max_summary_len=config["summary_len"],
        model_interface=model_interface,
        prompt_column_name=config["prompt_column_name"],
        hier_summ_column_name=config["hier_summ_column_name"],
        output_column_name=config["output_column_name"]

    )

    # Summarize data with progress bar
    df_summarized = generator.generate_summaries(df)

    # Save results
    df_summarized.to_csv(config["output"], index=False)
    print(f"Data saved to {config['output']}.\n")
