"""
Command-line interface for generating synthetic prompts.

This module provides a CLI for generating synthetic prompts from court rulings
and press releases using the Anthropic Claude API.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .generator import process_batch, generate_synthetic_prompt


def main():
    """Main entry point for the synthetic_prompts CLI."""
    parser = argparse.ArgumentParser(description="Generiere synthetische Prompts für Gerichtsurteile und Pressemitteilungen")
    
    # Hauptparameter
    parser.add_argument("--input", "-i", required=True, help="Pfad zur Eingabe-CSV-Datei mit Gerichtsurteilen und Pressemitteilungen")
    parser.add_argument("--output", "-o", required=True, help="Pfad für die Ausgabe-CSV-Datei")
    parser.add_argument("--checkpoint-dir", "-c", default="checkpoints", help="Verzeichnis für Checkpoints (Standard: checkpoints)")
    
    # Konfigurationsparameter
    parser.add_argument("--model", "-m", default="claude-3-7-sonnet-20250219", help="Zu verwendendes Claude-Modell")
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Anzahl der Elemente pro Batch (Standard: 5)")
    parser.add_argument("--start-idx", "-s", type=int, default=0, help="Startindex für die Verarbeitung (Standard: 0)")
    parser.add_argument("--save-interval", type=int, default=5, help="Speicherintervall für Checkpoints (Standard: 5)")
    parser.add_argument("--fix-errors", action="store_true", help="Fehlerhafte Einträge erneut verarbeiten")
    
    # API-Key-Parameter
    parser.add_argument("--api-key", help="Anthropic API-Schlüssel (alternativ über ANTHROPIC_API_KEY Umgebungsvariable)")
    parser.add_argument("--env-file", help="Pfad zur .env-Datei mit ANTHROPIC_API_KEY")
    
    # Single-Mode für einen einzelnen Test
    parser.add_argument("--test-single", action="store_true", 
                       help="Teste die Generierung mit einem einzelnen Eintrag aus dem Datensatz")
    parser.add_argument("--test-idx", type=int, default=0, 
                       help="Index des zu testenden Eintrags im Test-Modus (Standard: 0)")
    
    args = parser.parse_args()
    
    # Lade API-Schlüssel
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        # Standardmäßig nach .env-Datei im aktuellen Verzeichnis suchen
        load_dotenv()
    
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Fehler: Kein Anthropic API-Schlüssel gefunden. Bitte übergeben Sie ihn entweder über --api-key, "
              "setzen Sie die ANTHROPIC_API_KEY Umgebungsvariable, oder geben Sie eine .env-Datei mit --env-file an.")
        sys.exit(1)
    
    # Setze API-Schlüssel in Umgebungsvariable für spätere Verwendung
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Initialisiere den Claude API-Client
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Claude API-Client initialisiert für Modell: {args.model}")
    except Exception as e:
        print(f"Fehler bei der Initialisierung des Claude API-Clients: {e}")
        sys.exit(1)
    
    # Lade den Eingabe-Datensatz
    try:
        print(f"Lade Datensatz von {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Datensatz mit {len(df)} Einträgen geladen")
    except Exception as e:
        print(f"Fehler beim Laden des Datensatzes: {e}")
        sys.exit(1)
    
    # Prüfe, ob 'judgement' und 'summary' Spalten vorhanden sind
    required_columns = ['judgement', 'summary']
    for col in required_columns:
        if col not in df.columns:
            print(f"Fehler: Spalte '{col}' fehlt im Datensatz")
            sys.exit(1)
    
    # Erstelle Checkpoint-Verzeichnis
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"Checkpoints werden gespeichert in: {checkpoint_dir}")
    
    # Test-Modus für einen einzelnen Eintrag
    if args.test_single:
        if args.test_idx >= len(df):
            print(f"Fehler: Test-Index {args.test_idx} ist größer als die Datensatzgröße ({len(df)})")
            sys.exit(1)
        
        print(f"Test-Modus: Verarbeite Eintrag mit Index {args.test_idx}")
        row = df.iloc[args.test_idx]
        
        court_ruling = row['judgement']
        press_release = row['summary']
        
        # Kürze sehr lange Eingaben für die Anzeige
        court_ruling_display = court_ruling[:500] + "..." if len(court_ruling) > 500 else court_ruling
        press_release_display = press_release[:500] + "..." if len(press_release) > 500 else press_release
        
        print("\n=== Gerichtsurteil (gekürzt) ===")
        print(court_ruling_display)
        print("\n=== Pressemitteilung (gekürzt) ===")
        print(press_release_display)
        
        print("\nGeneriere synthetischen Prompt...")
        synthetic_prompt = generate_synthetic_prompt(court_ruling, press_release, client=client, model=args.model)
        
        print("\n=== Generierter synthetischer Prompt ===")
        print(synthetic_prompt)
        sys.exit(0)
    
    # Batch-Verarbeitung des gesamten Datensatzes
    print(f"Starte Batch-Verarbeitung mit Batch-Größe {args.batch_size} ab Index {args.start_idx}")
    print(f"Fix-Errors-Modus: {'Aktiviert' if args.fix_errors else 'Deaktiviert'}")
    
    # Extrahiere Ausgabe-Präfix aus dem Dateinamen
    output_prefix = Path(args.output).stem
    
    # Verarbeite den Datensatz
    result_df = process_batch(
        df, 
        batch_size=args.batch_size, 
        start_idx=args.start_idx, 
        save_interval=args.save_interval, 
        fix_errors=args.fix_errors,
        checkpoint_dir=checkpoint_dir,
        output_prefix=output_prefix,
        client=client
    )
    
    # Speichere das Endergebnis
    result_df.to_csv(args.output, index=False)
    print(f"Ergebnis mit {len(result_df)} Einträgen gespeichert unter {args.output}")
    
    # Gib Statistik aus
    success_count = result_df['synthetic_prompt'].notna().sum()
    error_count = sum(result_df['synthetic_prompt'].astype(str).str.contains("Fehler", na=False))
    print(f"Verarbeitung abgeschlossen. Erfolgreiche Prompts: {success_count}, Fehler: {error_count}")


if __name__ == "__main__":
    main()