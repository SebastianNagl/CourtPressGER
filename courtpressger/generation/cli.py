"""
Kommandozeilenschnittstelle für die Generierung von Pressemitteilungen.
"""
import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any, Optional

from .pipeline import LLMGenerationPipeline
from .teuken_generator import generate_with_teuken
from ..evaluation.models import create_model_config

def parse_args():
    """Parst die Kommandozeilenargumente für das Generierungs-Tool."""
    parser = argparse.ArgumentParser(description="Generierung von Pressemitteilungen aus Gerichtsurteilen mit LLMs")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Pfad zur Dataset-Datei (CSV oder JSON)")
    
    parser.add_argument("--output-dir", type=str, default="data/generation",
                        help="Verzeichnis für die Ausgabe der generierten Pressemitteilungen")
    
    parser.add_argument("--ruling-column", type=str, default="judgement",
                        help="Name der Spalte mit Gerichtsurteilen im Dataset")
    
    parser.add_argument("--prompt-column", type=str, default="synthetic_prompt",
                        help="Name der Spalte mit synthetischen Prompts im Dataset")
    
    parser.add_argument("--press-column", type=str, default="summary",
                        help="Name der Spalte mit Referenz-Pressemitteilungen im Dataset")
    
    parser.add_argument("--models-config", type=str, default="models/generation_config.json",
                        help="Pfad zur JSON-Konfigurationsdatei für die zu verwendenden Modelle")
    
    parser.add_argument("--model", type=str, default=None,
                        help="Verwende nur ein bestimmtes Modell (z.B. 'teuken-7b')")
    
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Anzahl der gleichzeitig zu verarbeitenden Einträge")
    
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Häufigkeit der Checkpoint-Speicherung (Anzahl der verarbeiteten Batches)")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximale Anzahl der zu verarbeitenden Einträge (für Tests)")
    
    parser.add_argument("--rate-limit-delay", type=float, default=1.0,
                        help="Verzögerung zwischen API-Aufrufen in Sekunden")
    
    # Zusätzliche Parameter für das Teuken-Modell
    parser.add_argument("--output-column", type=str, default="generated_pr_teuken",
                        help="Name der Spalte für die generierten Pressemitteilungen (nur für Teuken)")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperatur für die Generierung (0-1) (nur für Teuken)")
    
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximale Länge der generierten Texte (nur für Teuken)")
    
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p für Nucleus-Sampling (nur für Teuken)")
    
    parser.add_argument("--no-sample", action="store_true",
                        help="Deaktiviert Sampling (greedy decoding) (nur für Teuken)")
    
    args = parser.parse_args()
    return args

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Lädt das Dataset aus einer CSV- oder JSON-Datei."""
    if dataset_path.endswith(".csv"):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith(".json"):
        return pd.read_json(dataset_path, orient="records")
    else:
        raise ValueError(f"Unbekanntes Dateiformat: {dataset_path}")

def load_models_config(config_path: str, model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lädt die Modellkonfigurationen aus einer JSON-Datei.
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        model_filter: Optionaler Filtername für ein bestimmtes Modell
        
    Returns:
        Liste von Modellkonfigurationen
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    models_config = []
    for model_config in config_data["models"]:
        model_type = model_config.pop("type")
        name = model_config.pop("name")
        
        # Wenn ein Filter angegeben wurde, überspringe Modelle, die nicht dem Filter entsprechen
        if model_filter and name != model_filter:
            continue
        
        models_config.append(create_model_config(model_type, name, **model_config))
    
    return models_config

def main():
    """Haupteinstiegspunkt für das CLI-Tool."""
    args = parse_args()
    
    # Für das Teuken-Modell gibt es eine spezielle Implementierung
    if args.model == "teuken-7b":
        output_path = os.path.join(args.output_dir, f"{args.model}_results.csv")
        
        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Verwende die Teuken-spezifische Implementierung
        generate_with_teuken(
            dataset_path=args.dataset,
            output_path=output_path,
            model_path="models/teuken",
            ruling_column=args.ruling_column,
            prompt_column=args.prompt_column,
            output_column=args.output_column,
            batch_size=args.batch_size,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=not args.no_sample,
            top_p=args.top_p
        )
        
        print(f"\nTeuken-generierte Pressemitteilungen wurden in {output_path} gespeichert.")
        return
    
    # Für alle anderen Modelle oder wenn kein spezifisches Modell angegeben wurde
    # Dataset laden
    print(f"Lade Dataset aus {args.dataset}...")
    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset.head(args.limit)
    print(f"Dataset geladen: {len(dataset)} Einträge")
    
    # Überprüfen, ob alle benötigten Spalten vorhanden sind
    required_columns = [args.ruling_column, args.prompt_column, args.press_column]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Fehlende Spalten im Dataset: {', '.join(missing_columns)}")
    
    # Modellkonfigurationen laden
    print(f"Lade Modellkonfigurationen aus {args.models_config}...")
    models_config = load_models_config(args.models_config, args.model)
    
    if not models_config:
        if args.model:
            raise ValueError(f"Modell '{args.model}' nicht in der Konfiguration gefunden")
        else:
            raise ValueError("Keine Modelle in der Konfiguration gefunden")
    
    print(f"Modellkonfigurationen geladen: {len(models_config)} Modelle")
    
    # Generierungspipeline initialisieren und ausführen
    print("Starte Generierung...")
    pipeline = LLMGenerationPipeline(
        models_config, 
        output_dir=args.output_dir
    )
    
    results = pipeline.run_generation(
        dataset=dataset,
        prompt_column=args.prompt_column,
        ruling_column=args.ruling_column,
        reference_press_column=args.press_column,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        rate_limit_delay=args.rate_limit_delay
    )
    
    # Statistiken ausgeben
    print("\nGenerierungsstatistiken:")
    model_stats = results.groupby('model').agg({
        'id': 'count',
        'error': lambda x: x.str.len().gt(0).sum()
    }).rename(columns={'id': 'total', 'error': 'errors'})
    
    model_stats['success_rate'] = (model_stats['total'] - model_stats['errors']) / model_stats['total'] * 100
    
    print(model_stats)
    
    print(f"\nGenerierte Pressemitteilungen wurden in {args.output_dir} gespeichert.")

if __name__ == "__main__":
    main() 