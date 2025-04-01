"""
Implementierungen verschiedener LLMs für die Evaluierung der Pressemitteilungsgenerierung.
"""
import os
import requests
import json
from typing import Dict, Any, Optional, List
import concurrent.futures
from time import sleep
import numpy as np

class BaseModel:
    """Basisklasse für alle LLM-Implementierungen."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
    
    def generate(self, ruling: str, prompt: str) -> str:
        """
        Generiert eine Pressemitteilung basierend auf dem Gerichtsurteil und der Prompt.
        
        Args:
            ruling: Das Gerichtsurteil als Text
            prompt: Die synthetische Prompt
            
        Returns:
            Die generierte Pressemitteilung
        """
        raise NotImplementedError("Muss von abgeleiteten Klassen implementiert werden")
    
    def batch_generate(self, rulings: List[str], prompts: List[str], 
                      max_workers: int = 5, backoff_factor: float = 1.5, 
                      max_retries: int = 3) -> List[str]:
        """
        Generiert Pressemitteilungen für mehrere Urteile parallel.
        
        Args:
            rulings: Liste von Gerichtsurteilen
            prompts: Liste von Prompts (muss gleiche Länge wie rulings haben)
            max_workers: Maximale Anzahl paralleler Anfragen
            backoff_factor: Faktor für exponentielles Backoff bei Fehlern
            max_retries: Maximale Anzahl von Wiederholungsversuchen bei Fehlern
            
        Returns:
            Liste der generierten Pressemitteilungen
        """
        assert len(rulings) == len(prompts), "Anzahl der Urteile und Prompts muss übereinstimmen"
        
        results = [None] * len(rulings)
        
        def _process_item(idx):
            retries = 0
            while retries <= max_retries:
                try:
                    result = self.generate(rulings[idx], prompts[idx])
                    return idx, result, None
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        return idx, None, str(e)
                    sleep_time = backoff_factor ** retries
                    sleep(sleep_time)
            
            return idx, None, "Maximale Anzahl von Wiederholungsversuchen erreicht"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_item, i) for i in range(len(rulings))]
            
            for future in concurrent.futures.as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Fehler bei Index {idx}: {error}")
                results[idx] = result
        
        return results


class OpenAIModel(BaseModel):
    """Implementierung für OpenAI-Modelle (GPT-3.5, GPT-4 etc.)."""
    
    def __init__(self, name: str, model_name: str, api_key: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: int = 2048, **kwargs):
        """
        Initialisiert das OpenAI-Modell.
        
        Args:
            name: Name für dieses Modellinstanz
            model_name: Name des OpenAI-Modells (z.B. "gpt-3.5-turbo", "gpt-4")
            api_key: OpenAI API-Schlüssel (falls nicht gesetzt, wird OPENAI_API_KEY aus Umgebung verwendet)
            temperature: Temperaturparameter für die Generierung (0.0-1.0)
            max_tokens: Maximale Anzahl von Tokens in der Ausgabe
            **kwargs: Weitere Parameter für die OpenAI-API
        """
        super().__init__(name, model_name=model_name, temperature=temperature, 
                         max_tokens=max_tokens, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API-Schlüssel muss entweder als Parameter oder als Umgebungsvariable OPENAI_API_KEY angegeben werden")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI-Paket muss installiert sein. Führe 'uv add openai' aus.")
    
    def generate(self, ruling: str, prompt: str) -> str:
        """Generiert eine Pressemitteilung mit einem OpenAI-Modell."""
        system_prompt = "Du bist ein Experte für juristische Texte, der Pressemitteilungen basierend auf Gerichtsurteilen verfasst."
        
        # Erstellen einer kombinierten Prompt
        combined_prompt = f"{prompt}\n\nGerichtsurteil:\n{ruling}"
        
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        
        return response.choices[0].message.content


class HuggingFaceModel(BaseModel):
    """Implementierung für Modelle über die Hugging Face Inference API."""
    
    def __init__(self, name: str, model_id: str, api_key: Optional[str] = None, 
                api_url: Optional[str] = None, **kwargs):
        """
        Initialisiert das Hugging Face-Modell.
        
        Args:
            name: Name für diese Modellinstanz
            model_id: ID des Hugging Face-Modells
            api_key: Hugging Face API-Schlüssel (falls nicht gesetzt, wird HF_API_KEY aus Umgebung verwendet)
            api_url: Optionale benutzerdefinierte API-URL
            **kwargs: Weitere Parameter für die Hugging Face-API
        """
        super().__init__(name, model_id=model_id, **kwargs)
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API-Schlüssel muss entweder als Parameter oder als Umgebungsvariable HF_API_KEY angegeben werden")
        
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def generate(self, ruling: str, prompt: str) -> str:
        """Generiert eine Pressemitteilung mit einem Hugging Face-Modell."""
        combined_prompt = f"{prompt}\n\nGerichtsurteil:\n{ruling}"
        
        payload = {
            "inputs": combined_prompt,
            **{k: v for k, v in self.config.items() if k not in ["model_id", "api_url"]}
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Verarbeiten der Antwort basierend auf dem Format
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            
            return str(result)
        else:
            raise Exception(f"Fehler bei der Hugging Face API: {response.status_code} - {response.text}")


class LocalModel(BaseModel):
    """Implementierung für lokal gehostete Modelle (z.B. über vLLM, text-generation-webui etc.)."""
    
    def __init__(self, name: str, api_url: str, model_path: str = None, **kwargs):
        """
        Initialisiert das lokale Modell.
        
        Args:
            name: Name für diese Modellinstanz
            api_url: URL des lokal gehosteten API-Endpunkts
            model_path: Optional, Pfad zum lokalen Modell
            **kwargs: Weitere Parameter für die API-Anfrage
        """
        super().__init__(name, api_url=api_url, model_path=model_path, **kwargs)
        
        # Wenn ein lokales Modell angegeben wurde, lade es
        if model_path:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
    
    def generate(self, ruling: str, prompt: str) -> str:
        """Generiert eine Pressemitteilung mit einem lokal gehosteten Modell."""
        combined_prompt = f"{prompt}\n\nGerichtsurteil:\n{ruling}"
        
        # Wenn ein lokales Modell geladen wurde, verwende es direkt
        if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
            inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=self.config.get('max_length', 512))
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.get('max_length', 512),
                temperature=self.config.get('temperature', 0.7),
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ansonsten verwende die API
        payload = {
            "prompt": combined_prompt,
            **{k: v for k, v in self.config.items() if k not in ["api_url", "model_path"]}
        }
        
        response = requests.post(self.config["api_url"], json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extraktion der Antwort (Format kann je nach lokaler API variieren)
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("text", "")
            elif "generated_text" in result:
                return result["generated_text"]
            
            return str(result)
        else:
            raise Exception(f"Fehler bei der lokalen API: {response.status_code} - {response.text}")


def create_model_config(model_type: str, name: str, **kwargs) -> Dict[str, Any]:
    """
    Erzeugt eine Modellkonfiguration für die Evaluierungspipeline.
    
    Args:
        model_type: Typ des Modells ('openai', 'huggingface', 'local')
        name: Name für diese Modellinstanz
        **kwargs: Modellspezifische Parameter
        
    Returns:
        Modellkonfiguration als Dictionary
    """
    model_classes = {
        'openai': OpenAIModel,
        'huggingface': HuggingFaceModel,
        'local': LocalModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}. Muss einer von {list(model_classes.keys())} sein.")
    
    model_class = model_classes[model_type]
    model_instance = model_class(name=name, **kwargs)
    
    return {
        'name': name,
        'generator_fn': model_instance.generate,
        'model_instance': model_instance
    } 