"""
Kommandozeilenschnittstelle (CLI) für die Datensatzbereinigung.

Stellt einen einheitlichen Zugriff auf alle Bereinigungsmethoden
über eine Kommandozeilenschnittstelle bereit.
"""

import argparse
import sys
from pathlib import Path


def main():
    """
    Hauptfunktion der CLI, die die Argumente verarbeitet und die entsprechenden Skripte aufruft.
    """
    parser = argparse.ArgumentParser(description="Bereinigung des Court Press Datasets")
    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Befehle')
    
    # Common parser für gemeinsame Argumente
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--input", type=str, default="data/raw/cases_prs.csv",
                             help="Pfad zur Eingabedatei")
    common_parser.add_argument("--output", type=str, default="data/interim/cleaned.csv",
                             help="Pfad zur Ausgabedatei")
    common_parser.add_argument("--verbose", action="store_true",
                             help="Ausführliche Ausgabe")
    common_parser.add_argument("--no_plots", action="store_true",
                             help="Keine Plots erstellen")
    
    # 1. Rule-based Cleaning
    rule_parser = subparsers.add_parser('rule', parents=[common_parser],
                                       help='Regelbasierte Bereinigung durchführen')
    rule_parser.add_argument("--min_tokens", type=int, default=100,
                           help="Minimale Tokenanzahl")
    rule_parser.add_argument("--max_tokens", type=int, default=10000,
                           help="Maximale Tokenanzahl")
    rule_parser.add_argument("--min_similarity", type=float, default=0.1,
                           help="Minimale Ähnlichkeit")
    rule_parser.add_argument("--min_ratio", type=float, default=0.1,
                           help="Minimales Verhältnis Urteil/Pressemitteilung")
    rule_parser.add_argument("--max_ratio", type=float, default=10.0,
                           help="Maximales Verhältnis Urteil/Pressemitteilung")
    rule_parser.add_argument("--no_duplicate_check", action="store_true",
                           help="Keine Duplikate prüfen")
    rule_parser.add_argument("--no_filter", action="store_true",
                           help="Keine Filterung anwenden")
    
    # 2. Semantic Similarity
    similarity_parser = subparsers.add_parser('similarity', parents=[common_parser],
                                            help='Semantische Ähnlichkeiten berechnen')
    similarity_parser.add_argument("--method", type=str, default="tfidf",
                                 help="Methode zur Ähnlichkeitsberechnung (tfidf, bert)")
    similarity_parser.add_argument("--min_similarity", type=float, default=0.1,
                                 help="Minimale Ähnlichkeit")
    similarity_parser.add_argument("--max_features", type=int, default=10000,
                                 help="Maximale Anzahl an Features (für TF-IDF)")
    similarity_parser.add_argument("--ngram_range", type=str, default="1,3",
                                 help="N-Gram Bereich (für TF-IDF)")
    similarity_parser.add_argument("--use_gpu", action="store_true",
                                 help="GPU nutzen wenn verfügbar")
    
    # Argumente auswerten
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'rule':
        from courtpressger.data_cleaning.rule_based import main as rule_main
        # Konvertiere Namespace in Dictionary und entferne 'command'
        arg_dict = vars(args)
        arg_dict.pop('command')
        # Erstelle neue Argumentliste
        sys.argv = [sys.argv[0]]
        for key, value in arg_dict.items():
            # Überspringe False-Werte (von action=store_true) und None
            if value is not False and value is not None:
                if value is True:  # Für action=store_true
                    sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.append(f"--{key.replace('_', '-')}")
                    sys.argv.append(str(value))
        rule_main()
    
    elif args.command == 'similarity':
        from courtpressger.data_cleaning.semantic_similarity import main as similarity_main
        # Konvertiere Namespace in Dictionary und entferne 'command'
        arg_dict = vars(args)
        arg_dict.pop('command')
        # Erstelle neue Argumentliste
        sys.argv = [sys.argv[0]]
        for key, value in arg_dict.items():
            # Überspringe False-Werte (von action=store_true) und None
            if value is not False and value is not None:
                if value is True:  # Für action=store_true
                    sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.append(f"--{key.replace('_', '-')}")
                    sys.argv.append(str(value))
        similarity_main()
    
    else:
        print(f"Unbekannter Befehl: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
