"""
Kommandozeilenschnittstelle für die Evaluierung von LLMs zur Pressemitteilungsgenerierung.
"""
import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any

from .pipeline import LLMEvaluationPipeline
from .models import create_model_config
from .metrics import NLTK_AVAILABLE, BERT_SCORE_AVAILABLE

def parse_args():
    """Parst die Kommandozeilenargumente für das Evaluierungs-Tool."""
    parser = argparse.ArgumentParser(description="Evaluierung von LLMs für die Generierung von Pressemitteilungen")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Pfad zur Dataset-Datei (CSV oder JSON)")
    
    parser.add_argument("--output-dir", type=str, default="data/evaluation",
                        help="Verzeichnis für die Ausgabe der Evaluierungsergebnisse")
    
    parser.add_argument("--ruling-column", type=str, default="ruling",
                        help="Name der Spalte mit Gerichtsurteilen im Dataset")
    
    parser.add_argument("--prompt-column", type=str, default="synthetic_prompt",
                        help="Name der Spalte mit synthetischen Prompts im Dataset")
    
    parser.add_argument("--press-column", type=str, default="press_release",
                        help="Name der Spalte mit Referenz-Pressemitteilungen im Dataset")
    
    parser.add_argument("--models-config", type=str, required=True,
                        help="Pfad zur JSON-Konfigurationsdatei für die zu evaluierenden Modelle")
    
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Anzahl der gleichzeitig zu verarbeitenden Einträge")
    
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                        help="Häufigkeit der Checkpoint-Speicherung (Anzahl der verarbeiteten Batches)")
    
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximale Anzahl der zu verarbeitenden Einträge (für Tests)")
    
    parser.add_argument("--bert-score-model", type=str, default=None,
                        help="Name des Modells für BERTScore (falls nicht angegeben, wird BERTScore nicht berechnet)")
    
    parser.add_argument("--language", type=str, default="de",
                        help="Sprachcode für BERTScore (default: 'de' für Deutsch)")
    
    parser.add_argument("--metrics", type=str, nargs="+", 
                        default=["rouge", "bleu", "meteor", "bertscore"],
                        help="Zu berechnende Metriken: 'rouge', 'bleu', 'meteor', 'bertscore', 'alle'")
    
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
    
    # Verfügbarkeit der Metriken prüfen
    if "bleu" in args.metrics or "meteor" in args.metrics or "alle" in args.metrics:
        if not NLTK_AVAILABLE:
            print("Warnung: NLTK ist nicht installiert. BLEU und METEOR können nicht berechnet werden.")
            print("Verwende 'uv add nltk' und führe dann diese Python-Befehle aus:")
            print("  import nltk")
            print("  nltk.download('punkt')")
            print("  nltk.download('wordnet')")
            print("  nltk.download('omw-1.4')")
    
    if "bertscore" in args.metrics or "alle" in args.metrics:
        if not BERT_SCORE_AVAILABLE:
            print("Warnung: bert-score ist nicht installiert. BERTScore kann nicht berechnet werden.")
            print("Verwende 'uv add bert-score'.")
        elif not args.bert_score_model:
            print("Warnung: Kein Modell für BERTScore angegeben. BERTScore wird nicht berechnet.")
            print("Verwende --bert-score-model, um ein Modell anzugeben, z.B. 'microsoft/deberta-xlarge-mnli'.")
    
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
    
    # BERTScore-Modell setzen, falls in Metriken enthalten und Modell angegeben
    bert_score_model = None
    if ("bertscore" in args.metrics or "alle" in args.metrics) and args.bert_score_model:
        bert_score_model = args.bert_score_model
    
    # Evaluierungspipeline initialisieren und ausführen
    print("Starte Evaluierung...")
    pipeline = LLMEvaluationPipeline(
        models_config, 
        output_dir=args.output_dir,
        bert_score_model=bert_score_model,
        lang=args.language
    )
    
    results = pipeline.run_evaluation(
        dataset=dataset,
        prompt_column=args.prompt_column,
        ruling_column=args.ruling_column,
        reference_press_column=args.press_column,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Ergebnisse ausgeben
    print("\nEvaluierungsergebnisse:")
    for model_name, summary in results.items():
        print(f"\n{model_name}:")
        if "error" in summary:
            print(f"  Fehler: {summary['error']}")
            continue
        
        print(f"  Erfolgreiche Generierungen: {summary['successful_generations']}/{summary['total_samples']}")
        
        # ROUGE-Scores ausgeben
        if "rouge" in args.metrics or "alle" in args.metrics:
            print("\n  ROUGE-Scores:")
            if "avg_rouge1_fmeasure" in summary:
                print(f"    ROUGE-1 F1: {summary['avg_rouge1_fmeasure']:.4f}")
            if "avg_rouge2_fmeasure" in summary:
                print(f"    ROUGE-2 F1: {summary['avg_rouge2_fmeasure']:.4f}")
            if "avg_rougeL_fmeasure" in summary:
                print(f"    ROUGE-L F1: {summary['avg_rougeL_fmeasure']:.4f}")
        
        # BLEU-Scores ausgeben
        if ("bleu" in args.metrics or "alle" in args.metrics) and NLTK_AVAILABLE:
            print("\n  BLEU-Scores:")
            for i in range(1, 5):
                if f"avg_bleu{i}" in summary:
                    print(f"    BLEU-{i}: {summary[f'avg_bleu{i}']:.4f}")
        
        # METEOR-Score ausgeben
        if ("meteor" in args.metrics or "alle" in args.metrics) and NLTK_AVAILABLE:
            if "avg_meteor" in summary:
                print(f"\n  METEOR: {summary['avg_meteor']:.4f}")
        
        # BERTScore ausgeben
        if ("bertscore" in args.metrics or "alle" in args.metrics) and BERT_SCORE_AVAILABLE and bert_score_model:
            print("\n  BERTScore:")
            if "avg_bertscore_precision" in summary:
                print(f"    Precision: {summary['avg_bertscore_precision']:.4f}")
            if "avg_bertscore_recall" in summary:
                print(f"    Recall: {summary['avg_bertscore_recall']:.4f}")
            if "avg_bertscore_f1" in summary:
                print(f"    F1: {summary['avg_bertscore_f1']:.4f}")
    
    print(f"\nDetaillierte Ergebnisse wurden in {args.output_dir} gespeichert.")

if __name__ == "__main__":
    main() 