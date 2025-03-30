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
    Haupteinstiegspunkt für die Datenbereinigung.
    
    Bietet verschiedene Unterbefehle für die Bereinigung des Datensatzes,
    wie regelbasierte Filterung und Exportfunktionen.
    """
    # Hauptparser
    parser = argparse.ArgumentParser(
        description="Datenbereinigung für Gerichtsurteile-Datensatz")
    subparsers = parser.add_subparsers(dest="command", help="Unterbefehle")
    
    # Gemeinsame Argumente für alle Befehle
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--input", "-i", type=str, required=True,
                               help="Pfad zur Eingabe-CSV-Datei")
    common_parser.add_argument("--output_dir", "-o", type=str, default="outputs",
                               help="Ausgabeverzeichnis")
    common_parser.add_argument("--visualize", "-v", action="store_true",
                               help="Visualisierungen erstellen")
    common_parser.add_argument("--verbose", action="store_true",
                               help="Ausführliche Ausgabe")
    
    # 1. Rule-based filtering
    rule_parser = subparsers.add_parser('rule', parents=[common_parser],
                                       help='Regelbasierte Filterung anwenden')
    rule_parser.add_argument("--min_length", type=int, default=50,
                           help="Mindestlänge in Zeichen")
    rule_parser.add_argument("--min_date", type=str, default="2000-01-01",
                           help="Frühestes Datum (YYYY-MM-DD)")
    rule_parser.add_argument("--max_date", type=str, default="2023-12-31",
                           help="Spätestes Datum (YYYY-MM-DD)")
    rule_parser.add_argument("--min_ratio", type=float, default=0.01,
                           help="Minimales Verhältnis Urteil/Pressemitteilung")
    rule_parser.add_argument("--max_ratio", type=float, default=10.0,
                           help="Maximales Verhältnis Urteil/Pressemitteilung")
    rule_parser.add_argument("--no_duplicate_check", action="store_true",
                           help="Keine Duplikate prüfen")
    rule_parser.add_argument("--no_filter", action="store_true",
                           help="Keine Filterung anwenden")
    
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
    
    else:
        print(f"Unbekannter Befehl: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
