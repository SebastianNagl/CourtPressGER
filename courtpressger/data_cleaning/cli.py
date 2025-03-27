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
    parser = argparse.ArgumentParser(
        description="CourtPressGER Datensatzbereinigung CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Erstelle Subparser für die verschiedenen Befehle
    subparsers = parser.add_subparsers(
        dest='command', help='Bereinigungsmethode')

    # Gemeinsame Argumente für alle Befehle
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--input", "-i", type=str, required=True,
                               help="Pfad zur Eingabe-CSV-Datei")
    common_parser.add_argument("--output_dir", "-o", type=str, default="outputs",
                               help="Ausgabeverzeichnis")
    common_parser.add_argument("--visualize", "-v", action="store_true",
                               help="Visualisierungen erstellen")

    # 1. Regelbasierte Filterung
    rule_parser = subparsers.add_parser('rule_based', parents=[common_parser],
                                        help='Regelbasierte Filterung anwenden')
    # Keine zusätzlichen Argumente für regelbasierte Filterung

    # 2. Semantische Ähnlichkeit
    similarity_parser = subparsers.add_parser('similarity', parents=[common_parser],
                                              help='Semantische Ähnlichkeitsanalyse durchführen')
    similarity_parser.add_argument("--max_features", type=int, default=10000,
                                   help="Maximale Anzahl an TF-IDF Features")
    similarity_parser.add_argument("--no_preprocess", action="store_true",
                                   help="Textvorverarbeitung überspringen")

    # 3. ML-Klassifikation
    ml_parser = subparsers.add_parser('ml', parents=[common_parser],
                                      help='Machine Learning Klassifikation durchführen')
    ml_parser.add_argument("--max_features", type=int, default=5000,
                           help="Maximale Anzahl an TF-IDF Features")
    ml_parser.add_argument("--test_size", type=float, default=0.2,
                           help="Anteil der Testdaten")
    ml_parser.add_argument("--no_rule_based", action="store_true",
                           help="Regelbasierte Filter nicht verwenden")
    ml_parser.add_argument("--no_similarity", action="store_true",
                           help="Ähnlichkeitsfilter nicht verwenden")
    ml_parser.add_argument("--model_output", type=str,
                           help="Pfad zum Speichern des trainierten Modells")

    # 4. Clustering
    clustering_parser = subparsers.add_parser('clustering', parents=[common_parser],
                                              help='Clustering durchführen')
    clustering_parser.add_argument("--n_clusters", type=int, default=5,
                                   help="Anzahl der Cluster (für K-Means)")
    clustering_parser.add_argument("--n_components", type=int, default=50,
                                   help="Anzahl der PCA-Komponenten")
    clustering_parser.add_argument("--algorithm", type=str, default="kmeans",
                                   choices=["kmeans", "dbscan"],
                                   help="Clustering-Algorithmus")
    clustering_parser.add_argument("--no_pca", action="store_true",
                                   help="PCA nicht verwenden")
    clustering_parser.add_argument("--no_preprocess", action="store_true",
                                   help="Textvorverarbeitung überspringen")

    # 5. Kombinierte Pipeline
    combined_parser = subparsers.add_parser('combined', parents=[common_parser],
                                            help='Kombinierte Pipeline ausführen')
    combined_parser.add_argument("--min_votes", type=int, default=2,
                                 help="Minimale Anzahl an Methoden für Kombination")
    combined_parser.add_argument("--skip_rule_based", action="store_true",
                                 help="Regelbasierte Filterung überspringen")
    combined_parser.add_argument("--skip_similarity", action="store_true",
                                 help="Ähnlichkeitsberechnung überspringen")
    combined_parser.add_argument("--skip_ml", action="store_true",
                                 help="ML-Klassifikation überspringen")
    combined_parser.add_argument("--skip_clustering", action="store_true",
                                 help="Clustering überspringen")

    # Parse Argumente
    args = parser.parse_args()

    # Prüfe, ob ein Befehl angegeben wurde
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Führe entsprechenden Befehl aus
    if args.command == 'rule_based':
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

    elif args.command == 'ml':
        from courtpressger.data_cleaning.ml_classifier import main as ml_main
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
        ml_main()

    elif args.command == 'clustering':
        from courtpressger.data_cleaning.clustering import main as clustering_main
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
        clustering_main()

    elif args.command == 'combined':
        from courtpressger.data_cleaning.combined_pipeline import main as combined_main
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
        combined_main()

    else:
        print(f"Unbekannter Befehl: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
