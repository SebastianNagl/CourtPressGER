"""
Kommandozeilenschnittstelle für die Evaluierung von LLMs zur Pressemitteilungsgenerierung.
"""
import os
import argparse
import json
import pandas as pd
from typing import List, Dict, Any
import sys
import traceback

from .pipeline import LLMEvaluationPipeline
from .models import create_model_config
from .metrics import NLTK_AVAILABLE, BERT_SCORE_AVAILABLE


def parse_args():
    """Parst die Kommandozeilenargumente für das Evaluierungs-Tool."""
    parser = argparse.ArgumentParser(
        description="Evaluierung von LLMs für die Generierung von Pressemitteilungen")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Pfad zur Dataset-Datei (CSV, JSON, Parquet)")

    parser.add_argument("--output-dir", type=str, default="data/evaluation",
                        help="Verzeichnis für die Ausgabe der Evaluierungsergebnisse")

    parser.add_argument("--ruling-column", type=str, default="ruling",
                        help="Name der Spalte mit Gerichtsurteilen im Dataset")

    parser.add_argument("--prompt-column", type=str, default="synthetic_prompt",
                        help="Name der Spalte mit synthetischen Prompts im Dataset")

    parser.add_argument("--press-column", type=str, default="press_release",
                        help="Name der Spalte mit Referenz-Pressemitteilungen im Dataset")

    parser.add_argument("--source-text-column", type=str, default=None,
                        help="Name der Spalte mit dem Quelltext für sachliche Konsistenzmetriken (Standard: ruling-column)")

    # Gruppe für exklusive Argumente: Entweder Modellkonfiguration oder vorhandene Spalten
    model_source_group = parser.add_mutually_exclusive_group(required=True)
    model_source_group.add_argument("--models-config", type=str,
                                    help="Pfad zur JSON-Konfigurationsdatei für die zu evaluierenden Modelle (wenn neu generiert werden soll)")
    model_source_group.add_argument("--evaluate-existing-columns", action="store_true",
                                    help="Evaluiert vorhandene Spalten im Dataset als Modell-Outputs. Benötigt keine Modellkonfiguration.")

    parser.add_argument("--exclude-columns", type=str, nargs="*",
                        default=["id", "date", "summary", "judgement", "subset_name", "split_name",
                                 "is_announcement_rule", "matching_criteria", "synthetic_prompt", "ruling", "press_release"],
                        help="Spalten, die bei --evaluate-existing-columns ignoriert werden sollen")

    parser.add_argument("--batch-size", type=int, default=10,
                        help="Anzahl der gleichzeitig zu verarbeitenden Einträge")

    parser.add_argument("--checkpoint-freq", type=int, default=50,
                        help="Frequenz der Checkpoint-Speicherung")

    parser.add_argument("--limit", type=int, default=None,
                        help="Maximale Anzahl der zu verarbeitenden Einträge (für Tests)")

    parser.add_argument("--bert-score-model", type=str, default=None,
                        help="Name des Modells für BERTScore (z.B. 'models/eurobert')")

    parser.add_argument("--language", type=str, default="de",
                        help="Sprachcode für BERTScore (default: 'de' für Deutsch)")

    parser.add_argument("--metrics", type=str, nargs="+", default=["alle"],
                        choices=["alle", "rouge", "bleu", "meteor",
                                 "bertscore", "overlap", "qags", "factcc"],
                        help="Zu berechnende Metriken")

    # Neue Parameter für die Berichtsgenerierung
    parser.add_argument("--generate-report", action="store_true",
                        help="Generiert einen HTML-Bericht mit visualisierten Ergebnissen")

    parser.add_argument("--report-path", type=str, default=None,
                        help="Pfad für den HTML-Bericht (Standard: output-dir/report.html)")

    parser.add_argument("--enable-factual-consistency", action="store_true",
                        help="Aktiviert sachliche Konsistenzmetriken (QAGS, FactCC)")

    parser.add_argument("--enable-llm-as-judge", action="store_true",
                        help="Aktiviert die LLM-as-a-Judge Bewertung mit Claude 3.7 Sonnet")

    # Neue Parameter für die Berichtsgenerierung
    parser.add_argument("--generate-report", action="store_true",
                        help="Generiert einen HTML-Bericht mit visualisierten Ergebnissen")

    parser.add_argument("--report-path", type=str, default=None,
                        help="Pfad für den HTML-Bericht (Standard: output-dir/report.html)")

    return parser.parse_args()


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Lädt das Dataset aus einer Datei.

    Args:
        file_path: Pfad zur Datei (CSV, JSON, oder Parquet)

    Returns:
        DataFrame mit dem Dataset

    Raises:
        ValueError: Wenn das Dateiformat nicht unterstützt wird
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Nicht unterstütztes Dateiformat: {file_path}")


def load_models_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Lädt die Modellkonfigurationen aus einer JSON-Datei.

    Args:
        config_path: Pfad zur JSON-Konfigurationsdatei

    Returns:
        Liste mit Modellkonfigurationen
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    models_config = []

    for model_config in config["models"]:
        model_name = model_config["name"]
        model_type = model_config["type"]

        model_params = {"name": model_name}

        # Je nach Modelltyp die entsprechenden Parameter hinzufügen
        if model_type == "openai":
            model_params.update({
                "type": "openai",
                "model": model_config.get("model", "gpt-3.5-turbo"),
                "api_key": model_config.get("api_key", None),
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 1024)
            })
        elif model_type == "huggingface":
            model_params.update({
                "type": "huggingface",
                "model": model_config.get("model", "mistralai/Mistral-7B-v0.1"),
                "api_key": model_config.get("api_key", None),
                "api_url": model_config.get("api_url", "https://api-inference.huggingface.co/models/"),
                "temperature": model_config.get("temperature", 0.7),
                "max_new_tokens": model_config.get("max_new_tokens", 1024)
            })
        elif model_type == "local":
            model_params.update({
                "type": "local",
                "model_path": model_config.get("model_path", None),
                "temperature": model_config.get("temperature", 0.7),
                "max_new_tokens": model_config.get("max_new_tokens", 1024)
            })
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}")

        models_config.append(model_params)

    return models_config


def main():
    """Haupteinstiegspunkt für das CLI-Tool."""
    args = parse_args()

    # Verfügbarkeit der Metriken prüfen
    if "bleu" in args.metrics or "meteor" in args.metrics or "alle" in args.metrics:
        if not NLTK_AVAILABLE:
            print(
                "Warnung: NLTK ist nicht installiert. BLEU und METEOR können nicht berechnet werden.")
            print("Verwende 'uv add nltk' und führe dann diese Python-Befehle aus:")
            print("  import nltk")
            print("  nltk.download('punkt')")
            print("  nltk.download('wordnet')")
            print("  nltk.download('omw-1.4')")

    if "bertscore" in args.metrics or "alle" in args.metrics:
        if not BERT_SCORE_AVAILABLE:
            print(
                "Warnung: bert-score ist nicht installiert. BERTScore kann nicht berechnet werden.")
            print("Verwende 'uv add bert-score'.")
        elif not args.bert_score_model:
            print(
                "Warnung: Kein Modell für BERTScore angegeben. BERTScore wird nicht berechnet.")
            print(
                "Verwende --bert-score-model, um ein Modell anzugeben, z.B. 'microsoft/deberta-xlarge-mnli'.")

    # Dataset laden
    print(f"Lade Dataset aus {args.dataset}...")
    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset.head(args.limit)
    print(f"Dataset geladen: {len(dataset)} Einträge")

    # Überprüfen, ob alle benötigten Spalten vorhanden sind
    required_columns = [args.ruling_column,
                        args.prompt_column, args.press_column]
    missing_columns = [
        col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"Fehlende Spalten im Dataset: {', '.join(missing_columns)}")

    models_to_evaluate = []
    if args.evaluate_existing_columns:
        print("Evaluiere vorhandene Spalten im Dataset...")
        # Modellnamen aus Spalten extrahieren, exklusive bekannter Metadaten/Referenzspalten
        base_exclude = set(
            args.exclude_columns + [args.ruling_column, args.prompt_column, args.press_column])
        models_to_evaluate = [
            col for col in dataset.columns if col not in base_exclude]
        if not models_to_evaluate:
            raise ValueError(
                "Keine Modell-Spalten zur Evaluierung im Dataset gefunden (nach Ausschluss). Überprüfe --exclude-columns.")
        print(
            f"Folgende Spalten werden als Modelle evaluiert: {', '.join(models_to_evaluate)}")
    else:
        # Modellkonfigurationen laden (nur wenn --models-config verwendet wird)
        print(f"Lade Modellkonfigurationen aus {args.models_config}...")
        models_to_evaluate = load_models_config(args.models_config)
        print(
            f"Modellkonfigurationen geladen: {len(models_to_evaluate)} Modelle")

    # BERTScore-Modell setzen, falls in Metriken enthalten und Modell angegeben
    bert_score_model = None
    if ("bertscore" in args.metrics or "alle" in args.metrics) and args.bert_score_model:
        bert_score_model = args.bert_score_model

    # Evaluierungspipeline erstellen und ausführen
    pipeline = LLMEvaluationPipeline(
        models_to_evaluate,
        args.output_dir,
        bert_score_model=args.bert_score_model,
        lang=args.language
    )

    source_text_column = args.source_text_column or args.ruling_column

    results = pipeline.run_evaluation(
        dataset=dataset,
        prompt_column=args.prompt_column,
        ruling_column=args.ruling_column,
        reference_press_column=args.press_column,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        source_text_column=source_text_column,
        enable_factual_consistency=args.enable_factual_consistency,
        enable_llm_as_judge=args.enable_llm_as_judge
    )

    # Ausgabe
    print("\nEvaluierung abgeschlossen!")
    for model_name, summary in results.items():
        print(f"\n{model_name}:")
        if "error" in summary:
            print(f"  Fehler: {summary['error']}")
            continue

        print(
            f"  Erfolgreiche Generierungen: {summary['successful_generations']}/{summary['total_samples']}")

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
                print(
                    f"    Precision: {summary['avg_bertscore_precision']:.4f}")
            if "avg_bertscore_recall" in summary:
                print(f"    Recall: {summary['avg_bertscore_recall']:.4f}")
            if "avg_bertscore_f1" in summary:
                print(f"    F1: {summary['avg_bertscore_f1']:.4f}")

        # Überlappungsmetriken ausgeben
        if "overlap" in args.metrics or "alle" in args.metrics:
            print("\n  Überlappungsmetriken:")
            if "avg_keyword_overlap" in summary:
                print(
                    f"    Schlüsselwort-Überlappung: {summary['avg_keyword_overlap']:.4f}")
            if "avg_entity_overlap" in summary:
                print(
                    f"    Entitäts-Überlappung: {summary['avg_entity_overlap']:.4f}")
            # Entity-F1-Metriken anzeigen, wenn verfügbar
            if "avg_entity_precision" in summary and "avg_entity_recall" in summary and "avg_entity_f1" in summary:
                print(f"    Entity-F1-Score: {summary['avg_entity_f1']:.4f}")
                print(
                    f"    Entity-Precision: {summary['avg_entity_precision']:.4f}")
                print(f"    Entity-Recall: {summary['avg_entity_recall']:.4f}")
            if "avg_length_ratio" in summary:
                print(
                    f"    Längenverhältnis: {summary['avg_length_ratio']:.4f}")

        # QAGS ausgeben
        if ("qags" in args.metrics or "alle" in args.metrics) and args.enable_factual_consistency:
            print("\n  QAGS (Question Answering for evaluating Generated Summaries):")
            if "avg_qags_score" in summary:
                print(f"    Score: {summary['avg_qags_score']:.4f}")
            if "avg_qags_question_count" in summary:
                print(
                    f"    Durchschnittliche Fragenzahl: {summary['avg_qags_question_count']:.2f}")

        # FactCC ausgeben
        if ("factcc" in args.metrics or "alle" in args.metrics) and args.enable_factual_consistency:
            print("\n  FactCC (Factual Consistency Check):")
            if "avg_factcc_score" in summary:
                print(f"    Score: {summary['avg_factcc_score']:.4f}")
            if "avg_factcc_consistency_ratio" in summary:
                print(
                    f"    Konsistenz-Ratio: {summary['avg_factcc_consistency_ratio']:.4f}")

    print(
        f"\nDetaillierte Ergebnisse wurden in {args.output_dir} gespeichert.")

    # Optional einen Bericht mit Visualisierungen erstellen
    if args.generate_report:
        from .utils import create_report

        # Standard-Reportpfad setzen, falls nicht angegeben
        report_path = args.report_path if args.report_path else os.path.join(
            args.output_dir, "report.html")

        print(f"\nGeneriere Bericht mit Visualisierungen...")
        create_report(args.output_dir, report_path)
        print(f"Bericht wurde unter {report_path} gespeichert.")


if __name__ == "__main__":
    main()
