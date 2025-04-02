"""
Kommandozeilenschnittstelle für die Generierung von Pressemitteilungen.
"""
import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any, Optional

from .pipeline import LLMGenerationPipeline

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
    
    parser.add_argument("--checkpoint-freq", type=int, default=5,
                        help="Häufigkeit der Checkpoint-Speicherung (in Batches)")
    
    parser.add_argument("--rate-limit-delay", type=float, default=1.0,
                        help="Verzögerung zwischen API-Aufrufen in Sekunden")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit der zu verarbeitenden Datensätze (für Testzwecke)")
    
    return parser.parse_args()

def load_dataset(filepath: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Lädt das Dataset und beschränkt es optional auf eine bestimmte Anzahl von Einträgen.
    
    Args:
        filepath: Pfad zur Dataset-Datei
        limit: Optionale Begrenzung der Anzahl der zu ladenden Einträge
        
    Returns:
        DataFrame mit dem geladenen Dataset
    """
    # Dateityp bestimmen
    if filepath.endswith('.csv'):
        dataset = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        dataset = pd.read_json(filepath, orient='records')
    else:
        raise ValueError(f"Nicht unterstütztes Dateiformat: {filepath}")
    
    # Optional nur eine begrenzte Anzahl von Einträgen laden
    if limit is not None:
        dataset = dataset.head(limit)
    
    return dataset

def load_models_config(config_path: str, filter_model: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lädt die Modellkonfigurationen aus der angegebenen JSON-Datei.
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        filter_model: Wenn angegeben, nur dieses Modell zurückgeben
        
    Returns:
        Liste der Modellkonfigurationen
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    models_config = config.get('models', [])
    
    # Optional nach einem bestimmten Modell filtern
    if filter_model:
        models_config = [model for model in models_config if model.get('name') == filter_model]
    
    return models_config

def main():
    """Haupteinstiegspunkt für das CLI-Tool."""
    args = parse_args()
    
    print(f"Lade Dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.limit)
    print(f"Dataset geladen: {len(dataset)} Einträge")
    
    # Prüfen, ob erforderliche Spalten vorhanden sind
    required_columns = [args.ruling_column, args.prompt_column, args.press_column]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    
    if missing_columns:
        raise ValueError(f"Fehlende Spalten im Dataset: {', '.join(missing_columns)}")
    
    # Modellkonfigurationen laden
    print(f"Lade Modellkonfigurationen: {args.models_config}")
    models_config = load_models_config(args.models_config, args.model)
    
    if not models_config:
        if args.model:
            raise ValueError(f"Modell '{args.model}' nicht in der Konfiguration gefunden")
        else:
            raise ValueError("Keine Modelle in der Konfiguration gefunden")
    
    print(f"Modellkonfigurationen geladen: {len(models_config)} Modelle")
    
    # Generierungspipeline initialisieren und ausführen
    print("Starte Generierung mit Langchain-Pipeline...")
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
    # Prüfen, ob Ergebnisse vorhanden sind
    if results.empty:
        print("Keine Ergebnisse generiert. Überprüfen Sie die API-Keys und Modellverfügbarkeit.")
        return
    
    model_stats = results.groupby('model').agg({
        'id': 'count',
        'error': lambda x: x.str.len().gt(0).sum()
    }).rename(columns={'id': 'total', 'error': 'errors'})
    
    model_stats['success_rate'] = (model_stats['total'] - model_stats['errors']) / model_stats['total'] * 100
    
    print(model_stats)
    
    print(f"\nGenerierte Pressemitteilungen wurden in {args.output_dir} gespeichert.")

if __name__ == "__main__":
    main() 