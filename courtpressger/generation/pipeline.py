"""
Pipeline zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm
import time

class LLMGenerationPipeline:
    """
    Pipeline zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs.
    Die generierten Pressemitteilungen werden zusammen mit den tatsächlichen Pressemitteilungen gespeichert.
    """
    
    def __init__(self, models: List[Dict[str, Any]], output_dir: str = "data/generation"):
        """
        Initialisiert die Generierungspipeline.
        
        Args:
            models: Liste der zu verwendenden Modelle, jedes als Dict mit Namen und Generator-Funktion
            output_dir: Verzeichnis zum Speichern der generierten Pressemitteilungen
        """
        self.models = models
        self.output_dir = output_dir
        
        # Sicherstellen, dass das Ausgabeverzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)
    
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
        # Ergebnisdataframe erstellen
        result_df = pd.DataFrame()
        
        for model_config in self.models:
            model_name = model_config['name']
            model_fn = model_config['generator_fn']
            
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
                        generated_press = model_fn(ruling, prompt)
                        
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