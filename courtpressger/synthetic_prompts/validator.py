"""
CSV-Validator für Checkpoint-Dateien.

Dieses Modul bietet Funktionen zur Validierung von CSV-Checkpoint-Dateien,
um deren Integrität zu gewährleisten und häufige Fehler zu erkennen.
"""

import os
import glob
import pandas as pd
import logging
from pathlib import Path
import csv

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_csv_schema(file_path):
    """
    Überprüft, ob eine CSV-Datei dem erwarteten Schema entspricht.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        
    Returns:
        tuple: (bool, str) - (True, Nachricht) wenn die Datei gültig ist, sonst (False, Fehlermeldung)
    """
    try:
        # Versuche, die Datei als CSV zu laden
        df = pd.read_csv(file_path, on_bad_lines='warn')
        
        # Überprüfe, ob die erwarteten Spalten vorhanden sind
        expected_columns = [
            'id', 'date', 'summary', 'judgement', 'subset_name', 
            'split_name', 'is_announcement_rule', 'matching_criteria', 'synthetic_prompt'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            return False, f"Fehlende Spalten: {', '.join(missing_columns)}"
        
        # Überprüfe, ob die IDs eindeutig sind
        if df['id'].duplicated().any():
            duplicate_ids = df[df['id'].duplicated(keep=False)]['id'].unique()
            return False, f"Duplizierte IDs gefunden: {', '.join(map(str, duplicate_ids))}"
        
        # Überprüfe auf leere Werte in wichtigen Spalten
        important_columns = ['id', 'summary', 'judgement', 'synthetic_prompt']
        for col in important_columns:
            if df[col].isna().any() or (df[col] == '').any():
                missing_count = df[col].isna().sum() + (df[col] == '').sum()
                return False, f"Leere Werte in Spalte '{col}': {missing_count} Zeilen"
        
        # Überprüfe auf API-Fehlermeldungen in synthetic_prompt
        error_phrases = [
            "API-Fehler", 
            "Your credit balance is too low",
            "invalid_request_error",
            "Fehler bei der Generierung des Prompts"
        ]
        
        for phrase in error_phrases:
            if df['synthetic_prompt'].astype(str).str.contains(phrase).any():
                error_count = df['synthetic_prompt'].astype(str).str.contains(phrase).sum()
                return False, f"API-Fehlermeldungen '{phrase}' gefunden: {error_count} Zeilen"
        
        # Überprüfe auf unzulässige Zeichen oder Formatierungsprobleme
        for col in df.columns:
            if df[col].astype(str).str.contains('\0').any():  # Null-Bytes
                null_bytes_count = df[col].astype(str).str.contains('\0').sum()
                return False, f"Null-Bytes in Spalte '{col}' gefunden: {null_bytes_count} Zeilen"
        
        # Überprüfe auf übermäßig große Werte
        for col in df.columns:
            if df[col].astype(str).str.len().max() > 100000:  # 100KB als Schwellenwert
                max_length = df[col].astype(str).str.len().max()
                return False, f"Übermäßig große Werte in Spalte '{col}' gefunden: {max_length} Zeichen"
        
        return True, f"Die CSV-Datei ist gültig. {len(df)} Zeilen wurden überprüft."
        
    except Exception as e:
        return False, f"Fehler beim Validieren der CSV-Datei: {str(e)}"


def validate_all_checkpoints(directory="checkpoints", pattern="*_clean.csv"):
    """
    Überprüft alle Checkpoint-Dateien in einem Verzeichnis.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den Checkpoint-Dateien
        pattern (str): Glob-Pattern für die zu prüfenden Dateien
        
    Returns:
        dict: Dictionary mit Dateinamen als Schlüssel und (bool, str) als Werten
    """
    directory_path = Path(directory)
    
    # Finde alle passenden CSV-Dateien
    checkpoint_files = glob.glob(str(directory_path / pattern))
    
    logger.info(f"Gefundene Checkpoint-Dateien: {len(checkpoint_files)}")
    
    results = {}
    
    for file_path in checkpoint_files:
        file_name = Path(file_path).name
        logger.info(f"Überprüfe: {file_name}")
        
        is_valid, message = validate_csv_schema(file_path)
        results[file_name] = (is_valid, message)
        
        if is_valid:
            logger.info(f"✓ {file_name} ist gültig.")
        else:
            logger.warning(f"⚠️ {file_name} ist ungültig: {message}")
    
    # Zusammenfassung
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    logger.info(f"\nZusammenfassung: {valid_count}/{len(results)} Dateien sind gültig.")
    
    return results


def clean_csv_file(file_path, output_path=None):
    """
    Bereinigt eine CSV-Datei von häufigen Problemen.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        output_path (str, optional): Pfad für die bereinigte Datei. Wenn None, wird ein Name generiert.
        
    Returns:
        str: Pfad zur bereinigten Datei
    """
    if output_path is None:
        output_path = str(file_path).replace('.csv', '_cleaned.csv')
    
    logger.info(f"Bereinige CSV-Datei: {file_path}")
    
    try:
        # Versuche, die Datei zu lesen
        try:
            df = pd.read_csv(file_path, on_bad_lines='warn')
            logger.info(f"Datei erfolgreich als DataFrame geladen. {len(df)} Zeilen.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der CSV-Datei: {e}")
            logger.info("Versuche alternative Methode mit manueller Parsing...")
            
            # Versuche, mit erhöhtem Field-Size-Limit zu lesen
            csv.field_size_limit(int(1e9))  # Setze hohes Limit
            
            # Lese Header
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip().split(',')
            
            # Lese Zeilen mit CSV-Reader
            rows = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                next(reader)  # Überspringe Header
                for row in reader:
                    if len(row) == len(header):
                        rows.append(row)
            
            logger.info(f"Manuell {len(rows)} gültige Zeilen extrahiert.")
            df = pd.DataFrame(rows, columns=header)
        
        # Entferne doppelte IDs (behalte die erste)
        if 'id' in df.columns and df['id'].duplicated().any():
            duplicate_count = df['id'].duplicated().sum()
            logger.info(f"Entferne {duplicate_count} duplizierte IDs...")
            df = df.drop_duplicates(subset=['id'], keep='first')
        
        # Entferne Zeilen mit API-Fehlermeldungen
        error_phrases = [
            "API-Fehler", 
            "Your credit balance is too low",
            "invalid_request_error",
            "Fehler bei der Generierung des Prompts"
        ]
        
        initial_len = len(df)
        for phrase in error_phrases:
            if 'synthetic_prompt' in df.columns:
                error_mask = df['synthetic_prompt'].astype(str).str.contains(phrase, na=False)
                if error_mask.any():
                    error_count = error_mask.sum()
                    logger.info(f"Entferne {error_count} Zeilen mit '{phrase}'...")
                    df = df[~error_mask]
        
        removed_count = initial_len - len(df)
        if removed_count > 0:
            logger.info(f"Insgesamt {removed_count} Zeilen mit Fehlern entfernt.")
        
        # Überprüfe auf fehlende Werte in wichtigen Spalten
        important_columns = ['id', 'summary', 'judgement', 'synthetic_prompt']
        for col in important_columns:
            if col in df.columns:
                missing_mask = df[col].isna() | (df[col] == '')
                if missing_mask.any():
                    missing_count = missing_mask.sum()
                    logger.warning(f"Warnung: {missing_count} Zeilen mit fehlenden Werten in '{col}'.")
        
        # Speichere die bereinigte Datei mit verbesserten Sicherheitsoptionen
        df.to_csv(
            output_path,
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            doublequote=True,
            lineterminator='\n'
        )
        
        logger.info(f"Bereinigte Datei gespeichert: {output_path} mit {len(df)} Zeilen.")
        return output_path
    
    except Exception as e:
        logger.error(f"Fehler beim Bereinigen der CSV-Datei: {e}")
        raise


def main():
    """Hauptfunktion für CLI-Nutzung."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV-Validator für Checkpoint-Dateien")
    subparsers = parser.add_subparsers(dest="command", help="Befehle")
    
    # Validate-Befehl
    validate_parser = subparsers.add_parser("validate", help="Validiere CSV-Dateien")
    validate_parser.add_argument("--dir", "-d", default="checkpoints", help="Verzeichnis mit Checkpoint-Dateien")
    validate_parser.add_argument("--pattern", "-p", default="*_clean.csv", help="Glob-Pattern für zu prüfende Dateien")
    validate_parser.add_argument("--file", "-f", help="Einzelne Datei validieren (Optional)")
    
    # Clean-Befehl
    clean_parser = subparsers.add_parser("clean", help="Bereinige CSV-Dateien")
    clean_parser.add_argument("--file", "-f", required=True, help="Zu bereinigende CSV-Datei")
    clean_parser.add_argument("--output", "-o", help="Ausgabedatei (Optional)")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        if args.file:
            is_valid, message = validate_csv_schema(args.file)
            if is_valid:
                logger.info(f"✓ {args.file} ist gültig: {message}")
            else:
                logger.error(f"⚠️ {args.file} ist ungültig: {message}")
        else:
            validate_all_checkpoints(args.dir, args.pattern)
    
    elif args.command == "clean":
        output_path = clean_csv_file(args.file, args.output)
        logger.info(f"Bereinigung abgeschlossen: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 