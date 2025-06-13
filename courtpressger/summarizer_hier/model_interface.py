import os
import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer, pipeline
import tiktoken
import time
from openai import RateLimitError, OpenAIError
import requests

###############################################################################
# 1) ModelInterface definition
###############################################################################
class ModelInterface:
    """
    A simple interface for token counting and text generation.
    """
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError("count_tokens must be implemented by subclass")

    def generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError("generate_text must be implemented by subclass")


###############################################################################
# 2) Local Hugging Face Model Implementation
###############################################################################
class HuggingFaceLocalModel(ModelInterface):
    """
    Implementation of `ModelInterface` that uses a local Hugging Face model.
    """
    def __init__(
        self,
        model_name: str,
        do_sample: bool = True,
        top_p: float = 0.9,
        return_full_text=False,
        repetition_penalty: float = 1.2,
        torch_dtype=torch.bfloat16,
        device: str = "cuda:0",
        tokenizer_name: str = None
    ):
        HF_API_KEY = os.getenv("HF_API_KEY")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, token=HF_API_KEY, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.client = pipeline(
            "text-generation",
            model=model_name,
            return_full_text=return_full_text,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
            device=device  # Control which GPU/CPU to use
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        outputs = self.client(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        return outputs[0]['generated_text'].strip()


class DeepInfraModel(ModelInterface):
    """
    Implementation of `ModelInterface` that calls a DeepInfra-hosted model
    via their OpenAI-compatible /v1/chat/completions endpoint,
    retrying every 5 minutes on error. Adds truncation for Mistral models
    to enforce a 32,768-token context window, with logging of truncations.
    """
    MAX_MISTRAL_CONTEXT = 32700

    def __init__(
        self,
        model_name: str,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        tokenizer_name: str = None
    ):
        self.model_name = model_name
        self.api_key = os.getenv("DEEPINFRA_API_KEY")
        self.do_sample = do_sample
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.truncation_count = 0

        HF_API_KEY = os.getenv("HF_API_KEY")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            token=HF_API_KEY,
            trust_remote_code=True
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _truncate_if_mistral(self, messages, max_tokens):
        """
        If using a Mistral model and the combined messages exceed the context window,
        truncate the user content down to MAX_MISTRAL_CONTEXT tokens.
        """
        if "mistral" not in self.model_name.lower():
            return messages

        # Combine all message contents into one string
        full_text = "".join([msg.get("content", "") for msg in messages])
        token_count = self.count_tokens(full_text)
        if token_count + max_tokens <= self.MAX_MISTRAL_CONTEXT:
            return messages

        # Truncate to the first MAX_MISTRAL_CONTEXT tokens
        tokens = self.tokenizer.encode(full_text)
        truncated_tokens = tokens[: self.MAX_MISTRAL_CONTEXT-max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Replace with single user message
        truncated_msg = {"role": "user", "content": truncated_text}
        self.truncation_count += 1
        print(f"[DeepInfraModel] Truncated prompt to {self.MAX_MISTRAL_CONTEXT} tokens (occurrence #{self.truncation_count})")
        return [truncated_msg]

    def generate_text(self, prompt: list, max_tokens: int, temperature: float) -> str:
        """
        Calls the DeepInfra /v1/openai/chat/completions endpoint,
        retrying every 5 minutes if an error occurs. Applies truncation
        for Mistral models before sending.
        """
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Truncate messages if needed
        messages = self._truncate_if_mistral(prompt,max_tokens)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty
        }

        while True:
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

            except Exception as e:
                print(f"Error calling DeepInfra API: {e}. Retrying in 5 minutes...")
                time.sleep(5 * 60)  # wait 5 minutes then retry


###############################################################################
# 4) OpenAI Model Implementation (no organization ID)
###############################################################################
class OpenAIModel(ModelInterface):
    """
    Implementation of `ModelInterface` that calls the OpenAI ChatCompletion endpoint,
    but will pause and retry if it hits rate limits.
    """

    def __init__(self,
                 model_name: str = "gpt-4o",
                 do_sample: bool = True,
                 top_p: float = 0.9,
                 tokenizer_name: str = None,
                 max_retries: int = 5,
                 backoff_factor: float = 1.0):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)
        self.do_sample = do_sample
        self.top_p = top_p

        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # For token counting with tiktoken
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text.split())

    def generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        attempt = 0
        backoff = self.backoff_factor

        while True:
            try:
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                )
                return resp.output_text.strip()

            except RateLimitError as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise

                # if OpenAI returns a Retry-After header, respect it
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")
                sleep_secs = float(retry_after) if retry_after else backoff

                print(f"🔄 Rate limit hit; sleeping {sleep_secs}s before retry #{attempt}")
                time.sleep(sleep_secs)
                backoff *= 2  # exponential back‑off

            except OpenAIError as e:
                # any other OpenAI error: bubble up or handle as you see fit
                raise RuntimeError(f"OpenAI API error: {e}") from e


###############################################################################
# 5) get_model_interface: Return appropriate ModelInterface based on config
###############################################################################
def get_model_interface(config: dict, device: str = "cuda:0") -> ModelInterface:
    """
    Return an appropriate model interface instance based on config fields.

    Example of the config:
      {
        "model_type": "huggingface_local",
        "model": "some/hf-model",
        "do_sample": true,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "torch_dtype": "bfloat16",
        "api_key": "<your_api_key_if_needed>"
      }
    """
    model_type = config.get("model_type", "huggingface_local")
    print(f"Model type: {model_type}")
    print(f"Model: {config['model']}")
    if model_type == "huggingface_local":
        return HuggingFaceLocalModel(
            model_name=config["model"],
            do_sample=config.get("do_sample", True),
            top_p=config.get("top_p", 0.9),
            return_full_text=config.get("return_full_text", False),
            repetition_penalty=config.get("repetition_penalty", 1.2),
            torch_dtype=torch.bfloat16 if config.get("torch_dtype","bfloat16")=="bfloat16" else torch.float16,
            device=device,
            tokenizer_name=config.get("tokenizer_name")
        )
    elif model_type == "deep_infra":
        return DeepInfraModel(
            model_name=config["model"],
            do_sample=config.get("do_sample", True),
            top_p=config.get("top_p", 0.9),
            repetition_penalty=config.get("repetition_penalty", 1.2),
            tokenizer_name=config.get("tokenizer_name", None)
        )
    elif model_type == "openai":
        # If your config might have 'model' or 'model_name', handle both
        model_name = config.get("model") or config.get("model_name") or "gpt-3.5-turbo"
        return OpenAIModel(
            model_name=model_name,
            do_sample=config.get("do_sample", True),
            top_p=config.get("top_p", 0.9),
            tokenizer_name=config.get("tokenizer_name")
        )
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")