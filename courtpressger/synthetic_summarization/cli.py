import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import glob
import logging

import anthropic
from dotenv import load_dotenv

from .generator import process_batch


# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the synthetic_summarization CLI."""
    parser = argparse.ArgumentParser(description="Synthetische Prompt-Generierung für deutsche Gerichtsurteile und Pressemitteilungen. Dieses Modul bietet Funktionen für die Generierung, Verwaltung und Validierung von synthetischen Prompts unter Verwendung von LLMs.")
    
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    # Generate-Befehl 
    generate_parser = subparsers.add_parser("generate", help="Generiere synthetische Zusammenfassungen")
    generate_parser.add_argument("--input", "-i", required=True, help="Pfad zur Eingabe-CSV-Datei mit Gerichtsurteilen")
    generate_parser.add_argument("--output", "-o", required=True, help="Pfad für die Ausgabe-CSV-Datei")
    generate_parser.add_argument("--model", "-m", default="claude-3-7-sonnet-20250219", help="Zu verwendendes Claude-Modell")
    generate_parser.add_argument("--batch-size", "-b", type=int, default=10, help="Anzahl der Elemente pro Batch (Standard: 10)")

    args = parser.parse_args()

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Fehler: Kein Anthropic API-Schlüssel gefunden. Bitte übergeben Sie ihn entweder über --api-key, "
            "setzen Sie die ANTHROPIC_API_KEY Umgebungsvariable, oder geben Sie eine .env-Datei mit --env-file an.")
        sys.exit(1)
    
    os.environ["ANTHROPIC_API_KEY"] = api_key

    try:
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Claude API-Client initialisiert für Modell: {args.model}")
    except Exception as e:
        print(f"Fehler bei der Initialisierung des Claude API-Clients: {e}")
        sys.exit(1)

    try:
        print(f"Lade Datensatz von {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Datensatz mit {len(df)} Einträgen geladen")
    except Exception as e:
        print(f"Fehler beim Laden des Datensatzes: {e}")
        sys.exit(1)
    
    # Prüfe, ob 'judgement' und 'summary' Spalten vorhanden sind
    required_columns = ['judgement']
    for col in required_columns:
        if col not in df.columns:
            print(f"Fehler: Spalte '{col}' fehlt im Datensatz")
            sys.exit(1)

    print(f"Starte Batch-Verarbeitung mit Batch-Größe {args.batch_size}")


    result_df = process_batch(
            df, 
            batch_size=args.batch_size, 
            client=client
        )
    
    result_df.to_csv(args.output, index=False)
    print(f"Ergebnis mit {len(result_df)} Einträgen gespeichert unter {args.output}")
    
    # Gib Statistik aus
    success_count = result_df['synthetic_prompt'].notna().sum()
    error_count = sum(result_df['synthetic_prompt'].astype(str).str.contains("Fehler", na=False))
    print(f"Verarbeitung abgeschlossen. Erfolgreiche Prompts: {success_count}, Fehler: {error_count}")

if __name__ == "__main__":
    main()