"""
Kommandozeilenschnittstelle für die Generierung von Pressemitteilungen.
"""
import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from courtpressger.generation.pipeline import LLMGenerationPipeline

def parse_args():
    """Parst die Kommandozeilenargumente für das Generierungs-Tool."""
    parser = argparse.ArgumentParser(description="Generiere Pressemitteilungen aus Gerichtsurteilen")
    
    parser.add_argument("--input", type=str, required=True, help="Pfad zur Eingabedatei mit Gerichtsurteilen")
    
    parser.add_argument("--output", type=str, default="data/generation", help="Ausgabeverzeichnis für generierte Pressemitteilungen")
    
    parser.add_argument("--models", type=str, help="Pfad zur Modellkonfigurationsdatei (optional, verwendet standardmäßig models/generation_config.json)")
    
    parser.add_argument("--ignore-missing-columns", action="store_true", help="Ignoriere fehlende Spalten und nutze Standardwerte")
    
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
    
    # Lade die Modellkonfiguration
    if args.models:
        config_path = Path(args.models)
    else:
        config_path = Path("models/generation_config.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        models = config['models']
    
    # Initialisiere die Pipeline
    pipeline = LLMGenerationPipeline(models=models, output_dir=args.output)
    
    # Führe die Generierung durch
    # TODO: Implementiere die Generierung basierend auf der Eingabedatei
    
    print(f"Lade Dataset: {args.input}")
    dataset = load_dataset(args.input)
    print(f"Dataset geladen: {len(dataset)} Einträge")
    
    # Prüfen, ob erforderliche Spalten vorhanden sind
    required_columns = ['judgement', 'synthetic_prompt', 'summary']
    
    # Wenn ignore-missing-columns gesetzt ist, ersetze fehlende Spalten durch leere Werte
    if args.ignore_missing_columns:
        for col in required_columns:
            if col not in dataset.columns:
                print(f"Warnung: Spalte '{col}' fehlt im Dataset, wird mit leeren Werten erzeugt")
                dataset[col] = ""
    else:
        # Sonst prüfe wie bisher
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        
        if missing_columns:
            raise ValueError(f"Fehlende Spalten im Dataset: {', '.join(missing_columns)}")
    
    # Modellkonfigurationen laden
    print(f"Lade Modellkonfigurationen: {config_path}")
    models_config = load_models_config(str(config_path))
    
    if not models_config:
        raise ValueError("Keine Modelle in der Konfiguration gefunden")
    
    print(f"Modellkonfigurationen geladen: {len(models_config)} Modelle")
    
    # Generierungspipeline initialisieren und ausführen
    print("Starte Generierung mit Langchain-Pipeline...")
    results = pipeline.run_generation(
        dataset=dataset,
        prompt_column='synthetic_prompt',
        ruling_column='judgement',
        reference_press_column='summary',
        batch_size=10,
        checkpoint_freq=5,
        rate_limit_delay=1.0
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
    
    print(f"\nGenerierte Pressemitteilungen wurden in {args.output} gespeichert.")

if __name__ == "__main__":
    main() 