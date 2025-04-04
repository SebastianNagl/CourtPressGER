import os
import sys
import argparse

# Add project root to Python path to allow relative imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from courtpressger.evaluation.utils import create_report
except ImportError as e:
    print(f"Fehler beim Importieren von create_report: {e}", file=sys.stderr)
    print("Stellen Sie sicher, dass das Skript vom Projekt-Stammverzeichnis ausgeführt wird und die virtuelle Umgebung (.venv) aktiviert ist.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generiert einen HTML-Evaluierungsbericht aus vorhandenen Ergebnisdateien.")
    parser.add_argument("--results-dir", type=str, default="data/evaluation",
                        help="Verzeichnis mit den JSON-Ergebnisdateien der Evaluierung.")
    parser.add_argument("--output-file", type=str, default="reports/evaluation_report.html",
                        help="Pfad zur HTML-Ausgabedatei für den Bericht.")
    args = parser.parse_args()

    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    output_dir = os.path.dirname(args.output_file)
    if output_dir: # Nur erstellen, wenn nicht im Stammverzeichnis
        os.makedirs(output_dir, exist_ok=True)

    print(f"Erstelle Bericht aus Ergebnissen in '{args.results_dir}'...")
    try:
        create_report(results_dir=args.results_dir, output_path=args.output_file)
        # Kein print mehr hier, create_report gibt bereits eine Erfolgsmeldung aus
    except FileNotFoundError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        print(f"Stellen Sie sicher, dass die Evaluationspipeline (cli.py) erfolgreich durchgelaufen ist und Ergebnisdateien in '{args.results_dir}' existieren.", file=sys.stderr)
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}", file=sys.stderr)

if __name__ == "__main__":
    main() 