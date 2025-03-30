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
import glob
import logging

import anthropic
from dotenv import load_dotenv

from .generator import process_batch, generate_synthetic_prompt
from .sanitizer import (
    clean_checkpoint_file, verify_csv_file, fix_csv_format_errors, repair_csv_structure,
    sanitize_api_responses_in_csv, sanitize_all_files, clean_all_files, verify_all_files,
    validate_csv_schema, clean_csv_data
)


# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the synthetic_prompts CLI."""
    parser = argparse.ArgumentParser(description="Generiere synthetische Prompts für Gerichtsurteile und Pressemitteilungen")
    
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")
    
    # Generate-Befehl 
    generate_parser = subparsers.add_parser("generate", help="Generiere synthetische Prompts")
    generate_parser.add_argument("--input", "-i", required=True, help="Pfad zur Eingabe-CSV-Datei mit Gerichtsurteilen und Pressemitteilungen")
    generate_parser.add_argument("--output", "-o", required=True, help="Pfad für die Ausgabe-CSV-Datei")
    generate_parser.add_argument("--checkpoint-dir", "-c", default="checkpoints", help="Verzeichnis für Checkpoints (Standard: checkpoints)")
    generate_parser.add_argument("--model", "-m", default="claude-3-7-sonnet-20250219", help="Zu verwendendes Claude-Modell")
    generate_parser.add_argument("--batch-size", "-b", type=int, default=10, help="Anzahl der Elemente pro Batch (Standard: 10)")
    generate_parser.add_argument("--start-idx", "-s", type=int, default=0, help="Startindex für die Verarbeitung (Standard: 0)")
    generate_parser.add_argument("--save-interval", type=int, default=1, help="Speicherintervall für Checkpoints (Standard: 1, nach jedem Batch)")
    generate_parser.add_argument("--fix-errors", action="store_true", help="Fehlerhafte Einträge erneut verarbeiten")
    generate_parser.add_argument("--api-key", help="Anthropic API-Schlüssel (alternativ über ANTHROPIC_API_KEY Umgebungsvariable)")
    generate_parser.add_argument("--env-file", help="Pfad zur .env-Datei mit ANTHROPIC_API_KEY")
    generate_parser.add_argument("--test-single", action="store_true", help="Teste die Generierung mit einem einzelnen Eintrag aus dem Datensatz")
    generate_parser.add_argument("--test-idx", type=int, default=0, help="Index des zu testenden Eintrags im Test-Modus (Standard: 0)")
    
    # Clean-Befehl
    clean_parser = subparsers.add_parser("clean", help="Bereinige Checkpoint-Dateien von API-Fehlern")
    clean_parser.add_argument("--file", "-f", help="Pfad zur zu bereinigenden CSV-Datei (optional)")
    clean_parser.add_argument("--dir", "-d", default="checkpoints", help="Verzeichnis mit Checkpoint-Dateien (Standard: checkpoints)")
    clean_parser.add_argument("--output", "-o", help="Ausgabedatei für die Bereinigung (optional)")
    
    # Verify-Befehl
    verify_parser = subparsers.add_parser("verify", help="Überprüfe Checkpoint-Dateien auf API-Fehler")
    verify_parser.add_argument("--file", "-f", help="Pfad zur zu überprüfenden CSV-Datei (optional)")
    verify_parser.add_argument("--dir", "-d", default="checkpoints", help="Verzeichnis mit Checkpoint-Dateien (Standard: checkpoints)")
    verify_parser.add_argument("--pattern", "-p", default="*_clean.csv", help="Glob-Pattern für zu prüfende Dateien (Standard: *_clean.csv)")
    
    # Fix-Befehl
    fix_parser = subparsers.add_parser("fix", help="Behebe Formatierungsfehler in CSV-Dateien")
    fix_parser.add_argument("--file", "-f", required=True, help="Pfad zur zu reparierenden CSV-Datei")
    fix_parser.add_argument("--output", "-o", help="Pfad für die reparierte Datei (optional)")
    
    # Repair-Befehl
    repair_parser = subparsers.add_parser("repair", help="Repariere beschädigte CSV-Strukturen durch Neustrukturierung der Daten")
    repair_parser.add_argument("--file", "-f", required=True, help="Pfad zur zu reparierenden CSV-Datei")
    repair_parser.add_argument("--header-only", action="store_true", help="Extrahiere nur den Header und erstelle eine leere Datei (für schwer beschädigte Dateien)")
    
    # Validate-Befehl
    validate_parser = subparsers.add_parser("validate", help="Validiere CSV-Dateien auf Schema-Konformität")
    validate_parser.add_argument("--file", "-f", help="Einzelne Datei validieren (Optional)")
    validate_parser.add_argument("--dir", "-d", default="checkpoints", help="Verzeichnis mit Checkpoint-Dateien")
    validate_parser.add_argument("--pattern", "-p", default="*_clean.csv", help="Glob-Pattern für zu prüfende Dateien")
    
    # Sanitize-Befehl
    sanitize_parser = subparsers.add_parser("sanitize", help="Bereinige API-Antworten in CSV-Dateien")
    sanitize_parser.add_argument("--file", "-f", help="Pfad zur zu bereinigenden CSV-Datei (optional)")
    sanitize_parser.add_argument("--dir", "-d", default="checkpoints", help="Verzeichnis mit Checkpoint-Dateien (Standard: checkpoints)")
    sanitize_parser.add_argument("--output", "-o", help="Ausgabedatei für die bereinigte CSV (optional)")
    
    # Clean-CSV-Befehl
    clean_csv_parser = subparsers.add_parser("clean-csv", help="Bereinige CSV-Dateien von Problemen wie doppelten IDs, fehlenden Werten, etc.")
    clean_csv_parser.add_argument("--file", "-f", required=True, help="Pfad zur CSV-Datei")
    clean_csv_parser.add_argument("--output", "-o", help="Ausgabedatei (Optional)")
    
    args = parser.parse_args()
    
    if args.command == "generate":
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
    
    elif args.command == "clean":
        if args.file:
            # Bereinige eine einzelne Datei
            output_file = clean_checkpoint_file(args.file)
            print(f"Datei bereinigt und gespeichert als: {output_file}")
        else:
            # Bereinige alle Dateien im Verzeichnis
            cleaned_files = clean_all_files(args.dir)
            print(f"Alle Checkpoint-Dateien bereinigt. Ergebnisse: {len(cleaned_files)} Dateien verarbeitet.")
    
    elif args.command == "verify":
        if args.file:
            # Überprüfe eine einzelne Datei
            is_valid, message = verify_csv_file(args.file)
            print(f"Datei ist {'gültig' if is_valid else 'ungültig'}: {message}")
        else:
            # Überprüfe alle Dateien im Verzeichnis
            results = verify_all_files(args.dir, args.pattern)
            valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
            print(f"Alle Dateien sind {'gültig' if valid_count == len(results) else 'NICHT gültig'}.")
    
    elif args.command == "fix":
        # Behebe Formatierungsfehler in der Datei
        output_file = fix_csv_format_errors(args.file)
        print(f"Datei repariert und gespeichert als: {output_file}")
    
    elif args.command == "repair":
        # Repariere die CSV-Struktur
        output_file = repair_csv_structure(args.file, args.header_only)
        print(f"CSV-Struktur repariert und gespeichert als: {output_file}")
    
    elif args.command == "validate":
        if args.file:
            # Validiere eine einzelne Datei
            is_valid, errors = validate_csv_schema(args.file)
            if is_valid:
                print(f"✓ CSV-Datei ist gültig.")
            else:
                print(f"⚠️ CSV-Datei ist ungültig:")
                for error in errors:
                    print(f"  - {error}")
        else:
            # Validiere alle Dateien im Verzeichnis
            files = glob.glob(os.path.join(args.dir, args.pattern))
            valid_count = 0
            total_count = len(files)
            
            for file_path in files:
                print(f"\nValidiere: {file_path}")
                is_valid, errors = validate_csv_schema(file_path)
                if is_valid:
                    print(f"✓ Gültig.")
                    valid_count += 1
                else:
                    print(f"⚠️ Ungültig:")
                    for error in errors:
                        print(f"  - {error}")
            
            print(f"\nZusammenfassung: {valid_count}/{total_count} Dateien sind gültig.")
    
    elif args.command == "sanitize":
        if args.file:
            # Bereinige eine einzelne Datei
            output_file = sanitize_api_responses_in_csv(args.file)
            print(f"API-Antworten bereinigt und gespeichert als: {output_file}")
        else:
            # Bereinige alle Dateien im Verzeichnis
            sanitized_files = sanitize_all_files(args.dir)
            print(f"Bereinigung abgeschlossen: {len(sanitized_files)} Dateien verarbeitet.")
    
    elif args.command == "clean-csv":
        print(f"Bereinige CSV-Datei: {args.file}")
        output_file = clean_csv_data(args.file, args.output)
        print(f"Bereinigte Datei: {output_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()