"""
Pipeline zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs.
Verwendet Langchain für eine vereinfachte und leistungsfähige Integration verschiedener Modelle.
"""
import os
import json
import pandas as pd
import traceback
import sys
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
from pathlib import Path
from dotenv import load_dotenv

# Umgebungsvariablen aus .env-Datei laden
load_dotenv(verbose=True)

# Pfad zur .env-Datei ausgeben
print(f"Lade .env aus: {os.path.abspath('.env')}")

# Debug-Informationen zu API-Keys anzeigen
openai_key = os.getenv("OPENAI_API_KEY", "nicht gefunden")
openai_org = os.getenv("OPENAI_ORGANIZATION_ID", "nicht gefunden")
deepinfra_key = os.getenv("DEEPINFRA_API_KEY", "nicht gefunden")

print(f"OpenAI API-Key (gekürzt): {openai_key[:8]}...{openai_key[-4:] if len(openai_key) > 12 else ''}")
print(f"OpenAI Organization ID: {openai_org}")
print(f"DeepInfra API-Key: {'✓ gefunden' if deepinfra_key != 'nicht gefunden' else '✗ nicht gefunden'}")

# Langchain Importe
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain

# Transformers-Import für lokale Modelle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import für DeepInfra API, falls verfügbar
try:
    import requests
    import json as json_lib
    DEEPINFRA_AVAILABLE = True
except ImportError:
    DEEPINFRA_AVAILABLE = False
    print("⚠️ DeepInfra API nicht verfügbar. DeepInfra-Modelle werden nicht funktionieren.")

class LLMGenerationPipeline:
    """
    Pipeline zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs.
    Die generierten Pressemitteilungen werden zusammen mit den tatsächlichen Pressemitteilungen gespeichert.
    Verwendet Langchain für eine vereinfachte Integration verschiedener Modelle.
    """
    
    def __init__(self, models: List[Dict[str, Any]], output_dir: str = "data/generation"):
        """
        Initialisiert die Generierungspipeline.
        
        Args:
            models: Liste der zu verwendenden Modellkonfigurationen
            output_dir: Verzeichnis zum Speichern der generierten Pressemitteilungen
        """
        # Wenn models None ist, lade die Konfiguration aus der Datei
        if models is None:
            config_path = Path("models/generation_config.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                models = config['models']
        
        self.model_configs = models
        self.output_dir = output_dir
        self.chains = {}
        
        # Sicherstellen, dass das Ausgabeverzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)
        
        # Modelle initialisieren
        self._initialize_models()
        
        # API-Keys anzeigen (nur zu Test- und Entwicklungszwecken)
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Überprüft, ob die erforderlichen API-Keys vorhanden sind."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        hf_api_key = os.getenv("HF_API_KEY")
        deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
        
        if openai_api_key:
            print("✓ OpenAI API-Key gefunden")
        else:
            print("✗ OpenAI API-Key nicht gefunden - OpenAI-Modelle werden nicht funktionieren")
            
        if hf_api_key:
            print("✓ Hugging Face API-Key gefunden")
        else:
            print("✗ Hugging Face API-Key nicht gefunden - HF-Modelle könnten eingeschränkt sein")
            
        if deepinfra_api_key:
            print("✓ DeepInfra API-Key gefunden")
        else:
            print("✗ DeepInfra API-Key nicht gefunden - DeepInfra-Modelle werden nicht funktionieren")
    
    def _initialize_models(self):
        """Initialisiert die LLM-Chains für alle konfigurierten Modelle."""
        for model_config in self.model_configs:
            model_type = model_config['type']
            model_name = model_config['name']
            model_category = model_config.get('category', 'unbekannt')
            
            print(f"Initialisiere Modell: {model_name} (Typ: {model_type}, Kategorie: {model_category})")
            
            # Prompt-Template definieren (einheitlich für Vergleichbarkeit der Ergebnisse)
            # Der einheitliche Prompt besteht nur aus dem synthetischen Prompt und dem Ruling
            template = """{prompt}

Gerichtsurteil: {ruling}"""
            
            prompt = PromptTemplate.from_template(template)
            
            # Modell je nach Typ initialisieren
            if model_type == "openai":
                # API-Key aus der Umgebungsvariable abrufen
                api_key = os.getenv("OPENAI_API_KEY")
                
                # Prüfen, ob es sich um einen Project Key handelt (beginnt mit sk-proj-)
                is_project_key = api_key and api_key.startswith("sk-proj-")
                
                # Organisation ID aus der Umgebungsvariable abrufen, falls vorhanden
                organization_id = os.getenv("OPENAI_ORGANIZATION_ID")
                
                # Das Modell-Identifikator sicher aus der Konfiguration extrahieren
                # Wichtig: Sowohl 'model' als auch 'model_name' werden unterstützt
                # Bei Project Key soll 'model' verwendet werden
                if 'model' in model_config:
                    openai_model = model_config['model']
                elif 'model_name' in model_config:
                    openai_model = model_config['model_name']
                else:
                    openai_model = 'gpt-4o'  # Fallback
                    print(f"⚠️ Keine Modellbezeichnung gefunden für {model_name}, verwende {openai_model}")
                
                if is_project_key:
                    print(f"Project Key erkannt für {model_name}")
                    try:
                        # Bei Project Keys die direkte OpenAI API verwenden
                        from openai import OpenAI
                        
                        # OpenAI-Client mit Project Key initialisieren
                        openai_client = OpenAI(api_key=api_key)
                        
                        # Angepasste Funktion für API-Aufrufe
                        def openai_project_key_generate(ruling, prompt):
                            # Einheitlicher Prompt für alle Modelle (prompt zuerst, dann ruling)
                            full_prompt = f"{prompt}\n\nGerichtsurteil: {ruling}"
                            
                            try:
                                # API-Aufruf mit Project Key
                                response = openai_client.chat.completions.create(
                                    model=openai_model,
                                    messages=[{"role": "user", "content": full_prompt}],
                                    temperature=model_config.get('temperature', 0.7),
                                    max_tokens=model_config.get('max_tokens', 1024)
                                )
                                
                                # Rückgabe des generierten Texts
                                return response.choices[0].message.content.strip()
                            except Exception as e:
                                error_msg = f"Fehler bei OpenAI API-Aufruf: {str(e)}"
                                print(f"⚠️ {error_msg}")
                                return error_msg
                        
                        # Dummy-Chain erstellen, die unsere angepasste Funktion verwendet
                        from langchain.schema import runnable
                        
                        class ProjectKeyRunnable(runnable.Runnable):
                            def invoke(self, input_dict):
                                return openai_project_key_generate(
                                    input_dict.get("ruling", ""),
                                    input_dict.get("prompt", "")
                                )
                        
                        # Chain erstellen, die mit dem Project Key arbeitet
                        chain = ProjectKeyRunnable()
                    
                    except Exception as e:
                        print(f"⚠️ Fehler beim Initialisieren des OpenAI Project Key Clients: {str(e)}")
                        continue
                        
                else:
                    try:
                        # Standard-Implementierung für normale API-Keys
                        llm = ChatOpenAI(
                            model=openai_model,
                            temperature=model_config.get('temperature', 0.7),
                            max_tokens=model_config.get('max_tokens', 1024),
                            organization=organization_id if organization_id else None
                        )
                        
                        # LLM-Chain erstellen
                        chain = LLMChain(
                            llm=llm,
                            prompt=prompt,
                            output_parser=StrOutputParser()
                        )
                    except Exception as e:
                        print(f"⚠️ Fehler beim Initialisieren des OpenAI Standard Clients: {str(e)}")
                        continue
                    
            elif model_type == "huggingface":
                llm = HuggingFacePipeline.from_model_id(
                    model_id=model_config['model_id'],
                    task="text-generation",
                    pipeline_kwargs={
                        "max_length": model_config.get('max_length', 512),
                        "temperature": model_config.get('temperature', 0.7),
                        "do_sample": True
                    }
                )
                
                # LLM-Chain erstellen
                chain = LLMChain(
                    llm=llm,
                    prompt=prompt,
                    output_parser=StrOutputParser()
                )
                
            elif model_type == "local":
                try:
                    # Lokales Modell laden und initialisieren
                    model_path = model_config['model_path']
                    
                    print(f"Lade lokales Modell aus {model_path}...")
                    
                    # Tokenizer laden
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    
                    # Modell laden mit automatischer Gerätezuweisung
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    
                    # Text-Generations-Pipeline erstellen
                    text_gen_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=model_config.get('max_length', 1024),
                        temperature=model_config.get('temperature', 0.7),
                        do_sample=True,
                        truncation=True,
                        return_full_text=True,
                        pad_token_id=tokenizer.eos_token_id,
                        max_length=None  # Deaktiviere max_length
                    )
                    
                    # HuggingFacePipeline-Wrapper erstellen
                    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
                    
                    # LLM-Chain erstellen
                    chain = LLMChain(
                        llm=llm,
                        prompt=prompt,
                        output_parser=StrOutputParser()
                    )
                except Exception as e:
                    print(f"⚠️ Fehler beim Laden des lokalen Modells {model_name}: {str(e)}")
                    continue
            elif model_type == "deepinfra":
                if not DEEPINFRA_AVAILABLE:
                    print(f"⚠️ DeepInfra API nicht verfügbar - Überspringe Modell {model_name}")
                    continue
                
                api_key = os.getenv("DEEPINFRA_API_KEY")
                if not api_key:
                    print(f"⚠️ Kein DeepInfra API-Key gefunden - Überspringe Modell {model_name}")
                    continue
                
                try:
                    # OpenAI-Client mit DeepInfra-Konfiguration initialisieren
                    from openai import OpenAI
                    openai_client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepinfra.com/v1/openai"
                    )
                    
                    # Angepasste Funktion für DeepInfra API-Aufrufe
                    def deepinfra_generate(ruling, prompt):
                        # Einheitlicher Prompt für alle Modelle
                        full_prompt = f"{prompt}\n\nGerichtsurteil: {ruling}"
                        
                        try:
                            # API-Aufruf mit DeepInfra
                            response = openai_client.chat.completions.create(
                                model=model_config['model_id'],
                                messages=[{"role": "user", "content": full_prompt}],
                                temperature=model_config.get('temperature', 0.7),
                                max_tokens=model_config.get('max_tokens', 1024)
                            )
                            
                            return response.choices[0].message.content.strip()
                        except Exception as e:
                            print(f"⚠️ Fehler bei der DeepInfra API: {str(e)}")
                            return ""
                    
                    # Dummy-Chain erstellen, die unsere angepasste Funktion verwendet
                    from langchain.schema import runnable
                    
                    class DeepInfraRunnable(runnable.Runnable):
                        def invoke(self, input_dict):
                            return deepinfra_generate(
                                input_dict.get("ruling", ""),
                                input_dict.get("prompt", "")
                            )
                    
                    # Chain erstellen, die mit DeepInfra arbeitet
                    chain = DeepInfraRunnable()
                except Exception as e:
                    print(f"⚠️ Fehler beim Initialisieren des DeepInfra Clients: {str(e)}")
                    continue
            else:
                raise ValueError(f"Unbekannter Modelltyp: {model_type}")
            
            self.chains[model_name] = chain
            print(f"✓ Modell {model_name} initialisiert")
    
    def _extract_generated_text(self, full_output, input_text):
        """
        Extrahiert nur den generierten Text aus der vollständigen Ausgabe.
        
        Args:
            full_output: Die vollständige Ausgabe des Modells
            input_text: Der ursprüngliche Input
            
        Returns:
            str: Nur der vom Modell generierte Text
        """
        # Debug-Ausgabe
        print("\nDebug - Modellausgabe:", full_output)
        print("\nDebug - Input:", input_text)
        
        # Wenn die Ausgabe ein Dictionary ist (z.B. von der HuggingFace Pipeline)
        if isinstance(full_output, dict):
            return full_output.get('generated_text', '')
        elif isinstance(full_output, list) and len(full_output) > 0:
            if isinstance(full_output[0], dict):
                return full_output[0].get('generated_text', '')
            return full_output[0]
            
        # String-Ausgabe
        if isinstance(full_output, str):
            # Wenn der vollständige Output mit dem Input beginnt, entferne diesen
            if full_output.startswith(input_text):
                generated_text = full_output[len(input_text):].strip()
                if generated_text.startswith("\n"):
                    generated_text = generated_text.lstrip("\n")
                return generated_text
            
            # Nach "Gerichtsurteil:" im Text suchen und alles danach als generiert betrachten
            if "Gerichtsurteil:" in full_output:
                # Finde die Position des letzten Vorkommens von "Gerichtsurteil:"
                last_ruling_pos = full_output.rfind("Gerichtsurteil:")
                # Suche nach dem Ende des Urteils durch einen Doppel-Zeilenumbruch
                ruling_end = full_output.find("\n\n", last_ruling_pos)
                if ruling_end != -1:
                    # Nimm den Text nach dem Urteil
                    return full_output[ruling_end:].strip()
        
        # Wenn nichts funktioniert, gib den vollständigen Text zurück
        return str(full_output)
    
    def run_generation(self, dataset: pd.DataFrame, prompt_column: str, ruling_column: str, 
                     reference_press_column: str, batch_size: int = 10,
                     checkpoint_freq: int = 10, rate_limit_delay: float = 1.0) -> pd.DataFrame:
        """
        Führt die Generierung für alle Modelle durch.
        
        Args:
            dataset: DataFrame mit Urteilen, Prompts und Referenz-Pressemitteilungen
            prompt_column: Name der Spalte mit synthetischen Prompts
            ruling_column: Name der Spalte mit Gerichtsurteilen
            reference_press_column: Name der Spalte mit Referenz-Pressemitteilungen
            batch_size: Anzahl der gleichzeitig zu verarbeitenden Einträge
            checkpoint_freq: Häufigkeit der Checkpoint-Speicherung (in Batches)
            rate_limit_delay: Verzögerung zwischen API-Aufrufen in Sekunden
            
        Returns:
            DataFrame mit den generierten Pressemitteilungen und Referenzen
        """
        # Prüfen, ob überhaupt Modelle verfügbar sind
        if not self.chains:
            print("⚠️ Keine Modelle verfügbar! Bitte API-Keys konfigurieren oder Modelle installieren.")
            # Leeres DataFrame zurückgeben
            return pd.DataFrame(columns=['id', 'model', 'generated_press', 'reference_press', 'error', 'prompt'])
        
        # Ergebnisdataframe erstellen
        result_df = pd.DataFrame()
        
        for model_name, chain in self.chains.items():
            print(f"Generiere Pressemitteilungen mit Modell: {model_name}")
            
            # Checkpoint-Datei für dieses Modell
            checkpoint_path = os.path.join(self.output_dir, f"{model_name}_checkpoint.json")
            
            # Bereits vorhandene Ergebnisse laden, falls vorhanden
            existing_results = {}
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                
                print(f"Lade {len(existing_results)} bestehende Ergebnisse aus Checkpoint")
            
            # Neue Ergebnisse initialisieren
            new_results = {}
            
            # Durchlaufe alle Batches
            for batch_start in tqdm(range(0, len(dataset), batch_size), 
                                   desc=f"Generiere mit {model_name}"):
                batch_data = dataset.iloc[batch_start:batch_start + batch_size]
                
                for _, row in batch_data.iterrows():
                    idx = str(row.name)
                    
                    # Überspringen, wenn bereits Ergebnisse vorhanden sind
                    if idx in existing_results:
                        new_results[idx] = existing_results[idx]
                        continue
                    
                    ruling = row[ruling_column]
                    prompt = row[prompt_column]
                    reference = row[reference_press_column]
                    
                    # Pressemitteilung mit dem Modell generieren
                    try:
                        generated_press = chain.invoke({
                            "ruling": ruling, 
                            "prompt": prompt
                        })
                        
                        # Bei Bedarf extrahiere nur den generierten Teil
                        input_text = f"Gerichtsurteil: {ruling}\n\n{prompt}"
                        generated_press = self._extract_generated_text(generated_press, input_text)
                        
                        # Ergebnisse speichern
                        new_results[idx] = {
                            'generated_text': generated_press,
                            'reference_text': reference,
                            'prompt': prompt,
                            'ruling': ruling
                        }
                        
                        # Kurze Pause für API Rate Limiting
                        time.sleep(rate_limit_delay)
                    
                    except Exception as e:
                        print(f"Fehler bei der Verarbeitung von Index {idx}: {str(e)}")
                        new_results[idx] = {
                            'error': str(e),
                            'reference_text': reference,
                            'prompt': prompt,
                            'ruling': ruling
                        }
                
                # Regelmäßig Checkpoints speichern
                if ((batch_start + batch_size) // batch_size) % checkpoint_freq == 0:
                    # Kombiniere bestehende und neue Ergebnisse
                    combined_results = {**existing_results, **new_results}
                    with open(checkpoint_path, 'w', encoding='utf-8') as f:
                        json.dump(combined_results, f, ensure_ascii=False, indent=2)
            
            # Finalen Checkpoint speichern
            combined_results = {**existing_results, **new_results}
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, ensure_ascii=False, indent=2)
            
            # Ergebnisse als CSV speichern
            results_df = pd.DataFrame.from_dict(
                {idx: {
                    'id': idx,
                    'model': model_name,
                    'generated_press': result.get('generated_text', ''),
                    'reference_press': result.get('reference_text', ''),
                    'error': result.get('error', ''),
                    'prompt': result.get('prompt', '')
                } for idx, result in combined_results.items()},
                orient='index'
            )
            
            # CSV speichern
            results_csv_path = os.path.join(self.output_dir, f"{model_name}_results.csv")
            results_df.to_csv(results_csv_path, index=False)
            
            # Zum Gesamtergebnis hinzufügen
            if result_df.empty:
                result_df = results_df
            else:
                result_df = pd.concat([result_df, results_df], ignore_index=True)
        
        # Gesamt-CSV speichern
        all_results_path = os.path.join(self.output_dir, "all_models_results.csv")
        result_df.to_csv(all_results_path, index=False)
        
        return result_df 