"""
Kommandozeilenschnittstelle für die Generierung von Pressemitteilungen.
"""
import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any

from .pipeline import LLMGenerationPipeline
from ..evaluation.models import create_model_config

def parse_args():
    """Parst die Kommandozeilenargumente für das Generierungs-Tool."""
    parser = argparse.ArgumentParser(description="Generierung von Pressemitteilungen aus Gerichtsurteilen mit LLMs")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Pfad zur Dataset-Datei (CSV oder JSON)")
    
    parser.add_argument("--output-dir", type=str, default="data/generation",
                        help="Verzeichnis für die Ausgabe der generierten Pressemitteilungen")
    
    parser.add_argument("--ruling-column", type=str, default="ruling",
                        help="Name der Spalte mit Gerichtsurteilen im Dataset")
    
    parser.add_argument("--prompt-column", type=str, default="synthetic_prompt",
                        help="Name der Spalte mit synthetischen Prompts im Dataset")
    
    parser.add_argument("--press-column", type=str, default="press_release",
                        help="Name der Spalte mit Referenz-Pressemitteilungen im Dataset")
    
    parser.add_argument("--models-config", type=str, required=True,
                        help="Pfad zur JSON-Konfigurationsdatei für die zu verwendenden Modelle")
    
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Anzahl der gleichzeitig zu verarbeitenden Einträge")
    
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Häufigkeit der Checkpoint-Speicherung (Anzahl der verarbeiteten Batches)")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximale Anzahl der zu verarbeitenden Einträge (für Tests)")
    
    parser.add_argument("--rate-limit-delay", type=float, default=1.0,
                        help="Verzögerung zwischen API-Aufrufen in Sekunden")
    
    return parser.parse_args()

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Lädt das Dataset aus einer CSV- oder JSON-Datei.
    
    Args:
        file_path: Pfad zur Datei
        
    Returns:
        DataFrame mit den Daten
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, orient='records')
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Nicht unterstütztes Dateiformat: {file_path}. Unterstützt werden CSV, JSON und Parquet.")

def load_models_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Lädt die Modellkonfigurationen aus einer JSON-Datei.
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        
    Returns:
        Liste von Modellkonfigurationen
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    models_config = []
    for model_config in config_data["models"]:
        model_type = model_config.pop("type")
        name = model_config.pop("name")
        models_config.append(create_model_config(model_type, name, **model_config))
    
    return models_config

def main():
    """Haupteinstiegspunkt für das CLI-Tool."""
    args = parse_args()
    
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
    models_config = load_models_config(args.models_config)
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