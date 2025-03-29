"""
CSV-Sanitizer für synthetische Prompts.

Dieses Modul bündelt alle Funktionen zur Bereinigung, Validierung und Reparatur
von CSV-Dateien und API-Antworten in einem einheitlichen Interface.
"""

import os
import sys
import glob
import pandas as pd
import re
import csv
import logging
import shutil
from pathlib import Path

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standard-Fehlermeldungen/Muster in API-Antworten
ERROR_PHRASES = [
    "API-Fehler", 
    "Your credit balance is too low",
    "invalid_request_error",
    "Fehler bei der Generierung des Prompts"
]

# Regexp-Patterns für Fehler in API-Antworten
ERROR_PATTERNS = [
    r".*Fehler bei der Generierung des Prompts.*",
    r".*API-Fehler.*",
    r".*Your credit balance is too low.*",
    r".*invalid_request_error.*",
    r".*Fehler:.*",
    r".*Error:.*",
    r".*Exception:.*"
]

# Cleanup-Muster für API-Antworten
API_CLEANUP_PATTERNS = [
    # Entferne "here is the prompt" oder ähnliche Meta-Kommentare
    (r"^(Hier ist der Prompt:?|Here is the prompt:?|The prompt would be:?|Folgender Prompt könnte verwendet werden:?)\s*", "", re.IGNORECASE),
    # Entferne Markdown-Formatierung
    (r"```prompt\s*|\s*```$", "", 0),
    (r"```\s*|\s*```$", "", 0),
    # Entferne Anführungszeichen am Anfang/Ende, wenn sie alleine stehen
    (r'^\s*[\'"]|[\'"]\s*$', "", 0),
    # Entferne übermäßige Leerzeichen/Zeilenumbrüche am Anfang/Ende
    (r"^\s+|\s+$", "", 0)
]

# CSV-Validierungsfunktionen

def validate_csv_schema(file_path):
    """
    Überprüft, ob eine CSV-Datei dem erwarteten Schema entspricht.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        
    Returns:
        tuple: (bool, list) - (True, []) wenn die Datei gültig ist, sonst (False, [Fehlermeldungen])
    """
    try:
        # Versuche, die Datei als CSV zu laden
        df = pd.read_csv(file_path, on_bad_lines='warn')
        
        validation_errors = []
        
        # Überprüfe, ob die erwarteten Spalten vorhanden sind
        expected_columns = [
            'id', 'date', 'summary', 'judgement', 'subset_name', 
            'split_name', 'is_announcement_rule', 'matching_criteria', 'synthetic_prompt'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            validation_errors.append(f"Fehlende Spalten: {', '.join(missing_columns)}")
        
        # Überprüfe, ob die IDs eindeutig sind
        if 'id' in df.columns and df['id'].duplicated().any():
            duplicate_ids = df[df['id'].duplicated(keep=False)]['id'].unique()
            validation_errors.append(f"Doppelte IDs gefunden: {', '.join(map(str, duplicate_ids))}")
        
        # Überprüfe auf leere Werte in wichtigen Spalten
        important_columns = ['id', 'summary', 'judgement', 'synthetic_prompt']
        for col in important_columns:
            if col in df.columns and (df[col].isna().any() or (df[col] == '').any()):
                missing_count = df[col].isna().sum() + (df[col] == '').sum()
                validation_errors.append(f"Fehlende Werte in Spalte '{col}': {missing_count} Zeilen")
        
        # Überprüfe auf API-Fehlermeldungen in synthetic_prompt
        if 'synthetic_prompt' in df.columns:
            for phrase in ERROR_PHRASES:
                if df['synthetic_prompt'].astype(str).str.contains(phrase).any():
                    error_count = df['synthetic_prompt'].astype(str).str.contains(phrase).sum()
                    validation_errors.append(f"API-Fehlermeldungen '{phrase}' gefunden: {error_count} Zeilen")
        
        # Überprüfe auf unzulässige Zeichen oder Formatierungsprobleme
        for col in df.columns:
            if df[col].astype(str).str.contains('\0').any():  # Null-Bytes
                null_bytes_count = df[col].astype(str).str.contains('\0').sum()
                validation_errors.append(f"Null-Bytes in Spalte '{col}' gefunden: {null_bytes_count} Zeilen")
        
        # Überprüfe auf übermäßig große Werte
        for col in df.columns:
            if df[col].astype(str).str.len().max() > 100000:  # 100KB als Schwellenwert
                max_length = df[col].astype(str).str.len().max()
                validation_errors.append(f"Übermäßig große Werte in Spalte '{col}' gefunden: {max_length} Zeichen")
        
        if validation_errors:
            return False, validation_errors
        
        return True, []
        
    except Exception as e:
        return False, [f"Fehler beim Validieren der CSV-Datei: {str(e)}"]


def clean_csv_data(file_path, output_path=None):
    """
    Bereinigt eine CSV-Datei von häufigen Problemen wie doppelten IDs und fehlenden Werten.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        output_path (str, optional): Pfad für die bereinigte Datei. Wenn None, wird ein Name generiert.
        
    Returns:
        str: Pfad zur bereinigten Datei
    """
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    if output_path is None:
        output_path = file_path_str.replace('.csv', '_cleaned.csv')
    
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
        if 'synthetic_prompt' in df.columns:
            initial_len = len(df)
            for phrase in ERROR_PHRASES:
                error_mask = df['synthetic_prompt'].astype(str).str.contains(phrase, na=False)
                if error_mask.any():
                    error_count = error_mask.sum()
                    logger.info(f"Entferne {error_count} Zeilen mit '{phrase}'...")
                    df = df[~error_mask]
            
            removed_count = initial_len - len(df)
            if removed_count > 0:
                logger.info(f"Insgesamt {removed_count} Zeilen mit Fehlern entfernt.")
        
        # Entferne Zeilen mit fehlenden Werten in wichtigen Spalten
        important_columns = ['id', 'summary', 'judgement']
        for col in important_columns:
            if col in df.columns:
                missing_mask = df[col].isna() | (df[col] == '')
                if missing_mask.any():
                    missing_count = missing_mask.sum()
                    logger.info(f"Entferne {missing_count} Zeilen mit fehlenden Werten in '{col}'...")
                    df = df[~missing_mask]
        
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
        return file_path_str

# Bestehende Funktionen
def sanitize_api_response(response_text):
    """
    Bereinigt und filtert die Antwort der Anthropic API, um unerwünschte Elemente zu entfernen.
    
    Args:
        response_text (str): Die Rohtext-Antwort von der API
        
    Returns:
        str: Die bereinigte Antwort
    """
    if not response_text or not isinstance(response_text, str):
        return ""
    
    # Erkenne und entferne Fehler/Fehlermeldungen
    for pattern in ERROR_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            return f"Fehler bei der Generierung des Prompts: Fehlermuster erkannt"
    
    # Wende alle Clean-Ups an
    for pattern, replacement, flags in API_CLEANUP_PATTERNS:
        response_text = re.sub(pattern, replacement, response_text, flags=flags)
    
    # Prüfe auf ungewöhnlich kurze oder leere Antworten
    if len(response_text.strip()) < 10:
        return f"Fehler bei der Generierung des Prompts: Antwort zu kurz"
    
    # Prüfe auf ungewöhnlich lange Antworten (möglicherweise fehlerhaft)
    if len(response_text) > 5000:
        # Versuche, nur den relevanten Teil zu extrahieren oder kürze
        response_text = response_text[:5000] + "..."
    
    return response_text


def sanitize_api_responses_in_csv(file_path, output_suffix="_sanitized"):
    """
    Bereinigt alle API-Antworten in einer CSV-Datei mit der sanitize_api_response-Funktion.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        output_suffix (str): Suffix für die Ausgabedatei
        
    Returns:
        str: Pfad zur bereinigten Datei
    """
    logger.info(f"Bereinige API-Antworten in: {file_path}")
    
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    try:
        # Lese die CSV-Datei
        df = pd.read_csv(file_path, on_bad_lines='warn')
        logger.info(f"Datei enthält {len(df)} Zeilen")
        
        # Überprüfe, ob die 'synthetic_prompt' Spalte existiert
        if 'synthetic_prompt' not in df.columns:
            logger.warning("Keine 'synthetic_prompt' Spalte gefunden. Überspringe Bereinigung.")
            return file_path_str
        
        # Zähler für bereinigte Einträge
        sanitized_count = 0
        total_count = len(df)
        
        # Bereinige jede API-Antwort
        for idx, row in df.iterrows():
            if pd.notna(row['synthetic_prompt']):
                # Wende sanitize_api_response auf den Wert an
                original = str(row['synthetic_prompt'])
                sanitized = sanitize_api_response(original)
                
                # Setze den bereinigten Wert zurück in den DataFrame
                if original != sanitized:
                    df.at[idx, 'synthetic_prompt'] = sanitized
                    sanitized_count += 1
        
        logger.info(f"Bereinigte Einträge: {sanitized_count}/{total_count}")
        
        # Speichere die bereinigte Datei
        output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        logger.info(f"Bereinigte Datei gespeichert als: {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Fehler beim Bereinigen der API-Antworten: {e}")
        return file_path_str


def sanitize_all_files(directory="checkpoints", pattern="*.csv", output_suffix="_sanitized"):
    """
    Bereinigt alle API-Antworten in allen CSV-Dateien eines Verzeichnisses.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den CSV-Dateien
        pattern (str): Muster für die zu verarbeitenden Dateien
        output_suffix (str): Suffix für die Ausgabedateien
        
    Returns:
        list: Liste der Pfade zu den bereinigten Dateien
    """
    directory_path = Path(directory)
    
    # Finde alle CSV-Dateien im Verzeichnis
    csv_files = glob.glob(str(directory_path / pattern))
    
    logger.info(f"Gefundene CSV-Dateien: {len(csv_files)}")
    
    sanitized_files = []
    
    for file_path in csv_files:
        logger.info(f"\nVerarbeite: {file_path}")
        sanitized_file = sanitize_api_responses_in_csv(file_path, output_suffix)
        sanitized_files.append(sanitized_file)
        logger.info(f"Bereinigung abgeschlossen: {sanitized_file}")
    
    return sanitized_files


def clean_checkpoint_file(file_path, output_suffix="_clean"):
    """
    Bereinigt eine CSV-Datei durch Entfernen von Zeilen mit API-Fehlermeldungen.
    
    Args:
        file_path (str): Pfad zur zu bereinigenden CSV-Datei
        output_suffix (str): Suffix für die Ausgabedatei
        
    Returns:
        str: Pfad zur bereinigten Datei
    """
    logger.info(f"Bereinige Datei: {file_path}")
    
    # Konvertiere zu Path und dann zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    # Lese die CSV-Datei
    try:
        df = pd.read_csv(file_path, on_bad_lines='warn')
        logger.info(f"Original enthält {len(df)} Zeilen")
        
        # Überprüfe die 'synthetic_prompt' Spalte auf Fehlermeldungen
        if 'synthetic_prompt' in df.columns:
            for phrase in ERROR_PHRASES:
                error_mask = df['synthetic_prompt'].astype(str).str.contains(phrase, na=False)
                if error_mask.any():
                    error_count = error_mask.sum()
                    logger.info(f"Entferne {error_count} Zeilen mit Fehlermeldung '{phrase}'")
                    df = df[~error_mask]
            
            logger.info(f"Nach Bereinigung: {len(df)} Zeilen")
            
            # Schreibe die bereinigte Datei
            output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
            df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
            logger.info(f"Bereinigte Datei gespeichert als: {output_file}")
            return output_file
        else:
            logger.warning("Keine 'synthetic_prompt' Spalte gefunden. Überspringe Bereinigung.")
            return file_path_str
            
    except Exception as e:
        logger.error(f"Fehler beim Lesen der CSV-Datei als DataFrame: {e}")
        logger.info("Versuche alternative Methode...")
        
        # Alternative Methode: Datei zeilenweise lesen und API-Fehler entfernen
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        logger.info(f"Datei hat {len(lines)} Zeilen")
        
        # Filtere Zeilen mit API-Fehlermeldungen
        filtered_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            if any(error_phrase in line for error_phrase in ERROR_PHRASES):
                logger.info(f"Entferne Zeile {i+1}: {line[:100]}...")
                skip_next = True  # Überspringe auch die nächste Zeile
                continue
                
            filtered_lines.append(line)
        
        logger.info(f"Nach Bereinigung: {len(filtered_lines)} Zeilen")
        
        # Schreibe die bereinigte Datei
        output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        logger.info(f"Bereinigte Datei gespeichert als: {output_file}")
        return output_file


def clean_all_files(directory="checkpoints", pattern="*.csv", output_suffix="_clean"):
    """
    Bereinigt alle CSV-Dateien in einem Verzeichnis durch Entfernen von Zeilen mit API-Fehlermeldungen.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den CSV-Dateien
        pattern (str): Muster für die zu verarbeitenden Dateien
        output_suffix (str): Suffix für die Ausgabedateien
        
    Returns:
        list: Liste der Pfade zu den bereinigten Dateien
    """
    directory_path = Path(directory)
    
    # Finde alle CSV-Dateien im Verzeichnis
    csv_files = glob.glob(str(directory_path / pattern))
    
    logger.info(f"Gefundene CSV-Dateien: {len(csv_files)}")
    
    cleaned_files = []
    
    for file_path in csv_files:
        logger.info(f"\nVerarbeite: {file_path}")
        cleaned_file = clean_checkpoint_file(file_path, output_suffix)
        cleaned_files.append(cleaned_file)
    
    return cleaned_files


def verify_csv_file(file_path):
    """
    Überprüft, ob eine CSV-Datei API-Fehlermeldungen enthält.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        
    Returns:
        tuple: (bool, str) - (True, Nachricht) wenn keine API-Fehler gefunden werden
    """
    logger.info(f"Überprüfe Datei auf Fehler: {file_path}")
    
    # Verwende die validate_csv_schema-Funktion für die Validierung
    is_valid, errors = validate_csv_schema(file_path)
    
    # Überprüfe speziell auf API-Fehlermeldungen
    api_errors = [err for err in errors if "API-Fehlermeldungen" in err]
    
    if api_errors:
        error_message = f"Datei enthält API-Fehlermeldungen: {'; '.join(api_errors)}"
        logger.warning(f"⚠️ {error_message}")
        return False, error_message
    
    # Wenn keine API-Fehler, aber andere Validierungsfehler vorhanden sind
    if not is_valid:
        other_errors = '; '.join(errors)
        logger.warning(f"⚠️ Datei enthält andere Probleme: {other_errors}")
        return False, f"Datei enthält andere Probleme: {other_errors}"
    
    logger.info(f"✓ Datei ist fehlerfrei.")
    return True, "Datei enthält keine API-Fehler."


def verify_all_files(directory="checkpoints", pattern="*.csv"):
    """
    Überprüft alle CSV-Dateien in einem Verzeichnis auf Probleme.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den CSV-Dateien
        pattern (str): Muster für die zu überprüfenden Dateien
        
    Returns:
        dict: Dictionary mit Dateinamen als Schlüssel und (bool, str) als Werten
    """
    directory_path = Path(directory)
    
    # Finde alle passenden CSV-Dateien
    csv_files = glob.glob(str(directory_path / pattern))
    
    logger.info(f"Gefundene CSV-Dateien: {len(csv_files)}")
    
    results = {}
    
    for file_path in csv_files:
        file_name = Path(file_path).name
        logger.info(f"Überprüfe: {file_name}")
        
        is_valid, message = verify_csv_file(file_path)
        results[file_name] = (is_valid, message)
        
        if is_valid:
            logger.info(f"✓ {file_name} ist gültig.")
        else:
            logger.warning(f"⚠️ {file_name} ist ungültig: {message}")
    
    # Zusammenfassung
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    logger.info(f"\nZusammenfassung: {valid_count}/{len(results)} Dateien sind gültig.")
    
    return results


def fix_csv_format_errors(file_path, output_suffix="_fixed"):
    """
    Behebt Formatierungsfehler in einer CSV-Datei.
    
    Args:
        file_path (str): Pfad zur zu reparierenden CSV-Datei
        output_suffix (str): Suffix für die Ausgabedatei
        
    Returns:
        str: Pfad zur reparierten Datei
    """
    logger.info(f"Behebe Formatierungsfehler in: {file_path}")
    
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    try:
        # Setze maximale Field-Size für CSV-Reader
        csv.field_size_limit(sys.maxsize)
        
        # Lese die Datei mit dem CSV-Reader, um Formatierungsfehler zu finden
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            expected_fields = len(first_line.split(','))
            f.seek(0)  # Zurück zum Anfang
            
            reader = csv.reader(f)
            header = next(reader)  # Header lesen
            
            # Sammle gültige Zeilen
            valid_rows = [header]
            error_lines = []
            line_num = 1  # Header ist Zeile 1
            
            for row in reader:
                line_num += 1
                if len(row) != len(header):
                    error_lines.append((line_num, len(row), len(header)))
                    logger.warning(f"Zeile {line_num}: Erwartet {len(header)} Felder, gefunden {len(row)}")
                else:
                    valid_rows.append(row)
        
        # Formatierungsfehler gefunden?
        if error_lines:
            logger.info(f"Gefunden: {len(error_lines)} Zeilen mit Formatierungsfehlern")
            logger.info(f"Behalte {len(valid_rows)-1} gültige Zeilen (plus Header)")
            
            # Speichere die bereinigten Daten in einer neuen CSV-Datei
            output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
                writer.writerows(valid_rows)
            
            logger.info(f"Reparierte Datei gespeichert als: {output_file}")
            
            # Prüfe, ob die reparierte Datei korrekt gelesen werden kann
            try:
                df = pd.read_csv(output_file, on_bad_lines='warn')
                logger.info(f"Reparierte Datei erfolgreich geladen: {len(df)} Zeilen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der reparierten Datei: {e}")
        
        else:
            logger.info("Keine Formatierungsfehler gefunden.")
            output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
            shutil.copy(file_path, output_file)
            logger.info(f"Kopie gespeichert als: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Fehler beim Beheben von Formatierungsfehlern: {e}")
        return file_path_str


def repair_csv_structure(file_path, header_only=False, output_suffix="_repaired"):
    """
    Repariert eine beschädigte CSV-Struktur, indem die Datei neu strukturiert wird.
    
    Args:
        file_path (str): Pfad zur zu reparierenden CSV-Datei
        header_only (bool): Wenn True, nur Header extrahieren und leere Datei erstellen
        output_suffix (str): Suffix für die Ausgabedatei
        
    Returns:
        str: Pfad zur reparierten Datei
    """
    logger.info(f"Repariere CSV-Struktur: {file_path}")
    logger.info(f"Header-Only Modus: {'Aktiviert' if header_only else 'Deaktiviert'}")
    
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    try:
        # Versuch 1: Mit Pandas laden
        try:
            df = pd.read_csv(file_path, nrows=1)
            header = list(df.columns)
            logger.info(f"Header erfolgreich extrahiert: {', '.join(header)}")
            
            if header_only:
                # Nur Header-Modus: Erstelle leere Datei mit Header
                output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
                empty_df = pd.DataFrame(columns=header)
                empty_df.to_csv(output_file, index=False)
                logger.info(f"Leere Datei mit Header erstellt: {output_file}")
                return output_file
                
            # Versuche, den Rest zu laden (mit Fehlerbehandlung)
            df = pd.read_csv(file_path, on_bad_lines='skip')
            logger.info(f"Datei erfolgreich geladen: {len(df)} Zeilen")
            
            # Speichere die reparierte Datei
            output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
            df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
            logger.info(f"Reparierte Datei gespeichert als: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Fehler beim Laden mit pandas: {e}")
            logger.info("Versuche alternative Methode...")
        
        # Versuch 2: Manuelles Parsing
        try:
            # Erhöhe das Field-Size-Limit massiv
            csv.field_size_limit(sys.maxsize)
            
            # Öffne die Datei und extrahiere den Header
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header_line = f.readline().strip()
                header = [col.strip('"') for col in header_line.split(',')]
                logger.info(f"Header manuell extrahiert: {', '.join(header)}")
                
                if header_only:
                    # Nur Header-Modus: Erstelle leere Datei mit Header
                    output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
                    empty_df = pd.DataFrame(columns=header)
                    empty_df.to_csv(output_file, index=False)
                    logger.info(f"Leere Datei mit Header erstellt: {output_file}")
                    return output_file
                
                # Sammle gültige Datensätze
                valid_records = []
                error_count = 0
                success_count = 0
                
                # Zurück zum Anfang und Reader initialisieren
                f.seek(0)
                reader = csv.reader(f)
                next(reader)  # Header überspringen
                
                for i, row in enumerate(reader, start=2):  # Starte bei 2 (nach Header)
                    try:
                        if len(row) == len(header):
                            valid_records.append(row)
                            success_count += 1
                            if success_count % 10000 == 0:
                                logger.info(f"Verarbeitet: {success_count} gültige Datensätze")
                    except Exception as row_error:
                        error_count += 1
                        if error_count <= 10:  # Zeige nur die ersten 10 Fehler an
                            logger.warning(f"Fehler in Zeile {i}: {row_error}")
                
                logger.info(f"Verarbeitung abgeschlossen. Gültige Datensätze: {success_count}, Fehler: {error_count}")
                
                # Erstelle DataFrame aus den gültigen Datensätzen
                df = pd.DataFrame(valid_records, columns=header)
                
                # Speichere die reparierte Datei
                output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
                df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
                logger.info(f"Reparierte Datei gespeichert als: {output_file}")
                return output_file
                
        except Exception as e:
            logger.error(f"Fehler bei der manuellen Parsing-Methode: {e}")
        
        # Fallback: Erstelle eine leere Datei mit dem Standard-Header
        logger.warning("Alle Reparaturversuche fehlgeschlagen. Erstelle leere Datei mit Standard-Header.")
        standard_header = ['id', 'date', 'summary', 'judgement', 'subset_name', 'split_name', 
                          'is_announcement_rule', 'matching_criteria', 'synthetic_prompt']
        
        output_file = file_path_str.replace('.csv', f'{output_suffix}.csv')
        empty_df = pd.DataFrame(columns=standard_header)
        empty_df.to_csv(output_file, index=False)
        logger.info(f"Leere Datei mit Standard-Header erstellt: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Unbehandelbarer Fehler bei der CSV-Reparatur: {e}")
        return file_path_str 