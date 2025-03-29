"""
Modul zur Bereinigung von Checkpoint-Dateien.

Dieses Modul bietet Funktionen zur Bereinigung von API-Fehlermeldungen 
in den Checkpoint-Dateien, die während der Generierung von synthetischen Prompts 
auftreten können.
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path
import csv
import re

# Importiere die sanitize_api_response-Funktion aus dem generator-Modul
try:
    from .generator import sanitize_api_response
except ImportError:
    # Fallback-Implementation für den Fall, dass das Modul nicht importiert werden kann
    def sanitize_api_response(response_text):
        """
        Bereinigt und filtert die Antwort der Anthropic API.
        
        Fallback-Implementation für den Fall, dass die Funktion aus generator nicht importiert werden kann.
        """
        if not response_text or not isinstance(response_text, str):
            return ""
        
        # Erkenne und entferne Fehler/Fehlermeldungen
        error_patterns = [
            r".*Fehler bei der Generierung des Prompts.*",
            r".*API-Fehler.*",
            r".*Your credit balance is too low.*",
            r".*invalid_request_error.*",
            r".*Fehler:.*",
            r".*Error:.*",
            r".*Exception:.*"
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return f"Fehler bei der Generierung des Prompts: Fehlermuster erkannt"
        
        # Entferne Metainformationen, die manchmal von der API zurückgegeben werden
        cleanups = [
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
        
        # Wende alle Clean-Ups an
        for pattern, replacement, flags in cleanups:
            response_text = re.sub(pattern, replacement, response_text, flags=flags)
        
        # Prüfe auf ungewöhnlich kurze oder leere Antworten
        if len(response_text.strip()) < 10:
            return f"Fehler bei der Generierung des Prompts: Antwort zu kurz"
        
        # Prüfe auf ungewöhnlich lange Antworten (möglicherweise fehlerhaft)
        if len(response_text) > 5000:
            # Versuche, nur den relevanten Teil zu extrahieren oder kürze
            response_text = response_text[:5000] + "..."
        
        return response_text


def clean_checkpoint_file(file_path):
    """
    Bereinigt eine Checkpoint-Datei von API-Fehlermeldungen.
    
    Args:
        file_path (str): Pfad zur zu bereinigenden CSV-Datei
        
    Returns:
        str: Pfad zur bereinigten Datei
    """
    print(f"Bereinige Datei: {file_path}")
    
    # Konvertiere zu Path und dann zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    # Lese die CSV-Datei
    try:
        df = pd.read_csv(file_path, on_bad_lines='warn')
        print(f"Original enthält {len(df)} Zeilen")
        
        # Versuche die CSV-Datei zu bereinigen, indem fehlerhafte Einträge entfernt werden
        error_phrases = [
            "API-Fehler", 
            "Your credit balance is too low",
            "invalid_request_error",
            "Fehler bei der Generierung des Prompts"
        ]
        
        # Überprüfe die 'synthetic_prompt' Spalte auf Fehlermeldungen
        if 'synthetic_prompt' in df.columns:
            for phrase in error_phrases:
                error_mask = df['synthetic_prompt'].astype(str).str.contains(phrase, na=False)
                if error_mask.any():
                    error_count = error_mask.sum()
                    print(f"Entferne {error_count} Zeilen mit Fehlermeldung '{phrase}'")
                    df = df[~error_mask]
            
            print(f"Nach Bereinigung: {len(df)} Zeilen")
            
            # Schreibe die bereinigte Datei
            output_file = file_path_str.replace('.csv', '_clean.csv')
            df.to_csv(output_file, index=False)
            print(f"Bereinigte Datei gespeichert als: {output_file}")
            return output_file
        else:
            print("Keine 'synthetic_prompt' Spalte gefunden. Überspringe Bereinigung.")
            return file_path_str
            
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Datei als DataFrame: {e}")
        print("Versuche alternative Methode...")
        
        # Alternative Methode: Datei zeilenweise lesen und API-Fehler entfernen
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"Datei hat {len(lines)} Zeilen")
        
        # Filtere Zeilen mit API-Fehlermeldungen
        filtered_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            if any(error_phrase in line for error_phrase in [
                "API-Fehler", 
                "Your credit balance is too low",
                "invalid_request_error",
                "Fehler bei der Generierung des Prompts"
            ]):
                print(f"Entferne Zeile {i+1}: {line[:100]}...")
                skip_next = True  # Überspringe auch die nächste Zeile
                continue
                
            filtered_lines.append(line)
        
        print(f"Nach Bereinigung: {len(filtered_lines)} Zeilen")
        
        # Schreibe die bereinigte Datei
        output_file = file_path_str.replace('.csv', '_clean.csv')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        print(f"Bereinigte Datei gespeichert als: {output_file}")
        return output_file


def sanitize_api_responses_in_csv(file_path):
    """
    Bereinigt alle API-Antworten in einer CSV-Datei mit der sanitize_api_response-Funktion.
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        
    Returns:
        str: Pfad zur bereinigten Datei
    """
    print(f"Bereinige API-Antworten in: {file_path}")
    
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    try:
        # Lese die CSV-Datei
        df = pd.read_csv(file_path, on_bad_lines='warn')
        print(f"Datei enthält {len(df)} Zeilen")
        
        # Überprüfe, ob die 'synthetic_prompt' Spalte existiert
        if 'synthetic_prompt' not in df.columns:
            print("Keine 'synthetic_prompt' Spalte gefunden. Überspringe Bereinigung.")
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
        
        print(f"Bereinigte Einträge: {sanitized_count}/{total_count}")
        
        # Speichere die bereinigte Datei
        output_file = file_path_str.replace('.csv', '_sanitized.csv')
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        print(f"Bereinigte Datei gespeichert als: {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"Fehler beim Bereinigen der API-Antworten: {e}")
        return file_path_str


def sanitize_all_checkpoints(directory="checkpoints"):
    """
    Bereinigt alle API-Antworten in allen Checkpoint-Dateien eines Verzeichnisses.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den Checkpoint-Dateien
        
    Returns:
        list: Liste der Pfade zu den bereinigten Dateien
    """
    directory_path = Path(directory)
    
    # Finde alle CSV-Dateien im Checkpoint-Verzeichnis
    checkpoint_files = glob.glob(str(directory_path / "*.csv"))
    
    print(f"Gefundene Checkpoint-Dateien: {len(checkpoint_files)}")
    
    sanitized_files = []
    
    for file_path in checkpoint_files:
        print(f"\nVerarbeite: {file_path}")
        sanitized_file = sanitize_api_responses_in_csv(file_path)
        sanitized_files.append(sanitized_file)
        print(f"Bereinigung abgeschlossen: {sanitized_file}")
    
    return sanitized_files


def clean_all_checkpoints(directory="checkpoints"):
    """
    Bereinigt alle Checkpoint-Dateien in einem Verzeichnis.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den Checkpoint-Dateien
        
    Returns:
        list: Liste der Pfade zu den bereinigten Dateien
    """
    directory_path = Path(directory)
    
    # Finde alle CSV-Dateien im Checkpoint-Verzeichnis
    checkpoint_files = glob.glob(str(directory_path / "synthetic_prompts_*.csv"))
    
    # Filtere bereits bereinigte Dateien heraus
    checkpoint_files = [f for f in checkpoint_files if not (f.endswith("_clean.csv") or f.endswith("_original.csv"))]
    
    print(f"Gefundene Checkpoint-Dateien: {len(checkpoint_files)}")
    
    cleaned_files = []
    
    for file_path in checkpoint_files:
        print(f"\nVerarbeite: {file_path}")
        cleaned_file = clean_checkpoint_file(file_path)
        cleaned_files.append(cleaned_file)
        print(f"Bereinigung abgeschlossen: {cleaned_file}")
    
    return cleaned_files


def verify_checkpoint_file(file_path):
    """
    Überprüft, ob eine Checkpoint-Datei API-Fehlermeldungen enthält.
    
    Args:
        file_path (str): Pfad zur zu überprüfenden CSV-Datei
        
    Returns:
        bool: True, wenn keine Fehlermeldungen gefunden wurden, sonst False
    """
    print(f"Überprüfe Datei: {file_path}")
    
    # Fehlerhafte Phrasen, nach denen gesucht werden soll
    error_phrases = [
        "API-Fehler", 
        "Your credit balance is too low",
        "invalid_request_error",
        "Fehler bei der Generierung des Prompts"
    ]
    
    try:
        # Versuche die Datei als CSV zu lesen
        df = pd.read_csv(file_path, on_bad_lines='warn')
        print(f"Datei enthält {len(df)} Zeilen")
        
        # Überprüfe alle Spalten auf Fehlermeldungen
        has_errors = False
        for column in df.columns:
            for phrase in error_phrases:
                if df[column].astype(str).str.contains(phrase).any():
                    rows_with_errors = df[df[column].astype(str).str.contains(phrase)]
                    print(f"FEHLER: Spalte '{column}' enthält noch {len(rows_with_errors)} Zeilen mit '{phrase}'")
                    has_errors = True
        
        if not has_errors:
            print("✓ Keine API-Fehlermeldungen gefunden!")
            
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Datei: {e}")
        print("Versuche alternative Methode...")
        
        # Alternative Methode: Datei zeilenweise lesen
        has_errors = False
        error_lines = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                for phrase in error_phrases:
                    if phrase in line:
                        print(f"FEHLER: Zeile {i+1} enthält '{phrase}'")
                        error_lines.append((i+1, line[:100]))
                        has_errors = True
                        break
        
        if error_lines:
            print(f"Gefundene Fehlerzeilen ({len(error_lines)}):")
            for line_num, content in error_lines[:10]:  # Zeige nur die ersten 10 an
                print(f"  Zeile {line_num}: {content}...")
            
            if len(error_lines) > 10:
                print(f"  ... und {len(error_lines) - 10} weitere")
        
        if not has_errors:
            print("✓ Keine API-Fehlermeldungen gefunden!")
    
    return not has_errors


def verify_all_checkpoints(directory="checkpoints"):
    """
    Überprüft alle Checkpoint-Dateien in einem Verzeichnis.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den Checkpoint-Dateien
        
    Returns:
        bool: True, wenn alle Dateien fehlerfrei sind, sonst False
    """
    directory_path = Path(directory)
    
    # Finde alle bereinigten CSV-Dateien
    clean_files = glob.glob(str(directory_path / "*_clean.csv"))
    
    print(f"Gefundene bereinigte Dateien: {len(clean_files)}")
    
    all_clean = True
    
    for file_path in clean_files:
        print(f"\nÜberprüfe: {file_path}")
        if not verify_checkpoint_file(file_path):
            all_clean = False
    
    if all_clean:
        print("\n✓ Alle überprüften Dateien sind frei von API-Fehlermeldungen!")
    else:
        print("\n⚠️ Es wurden noch Dateien mit API-Fehlermeldungen gefunden!")
    
    return all_clean


def fix_csv_format_errors(file_path):
    """
    Versucht, Formatierungsfehler in einer CSV-Datei zu beheben.
    
    Args:
        file_path (str): Pfad zur zu reparierenden CSV-Datei
        
    Returns:
        str: Pfad zur reparierten Datei
    """
    print(f"Versuche Formatierungsfehler in {file_path} zu beheben...")
    
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    try:
        # Lese die CSV-Datei zeilenweise
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"Datei hat {len(lines)} Zeilen")
        
        # Nehme die erste Zeile als Header
        header = lines[0].strip() if lines else ""
        expected_fields = header.count(',') + 1
        
        print(f"Header hat {expected_fields} Felder")
        
        # Zeilenweise korrigieren
        fixed_lines = [header + '\n']  # Beginne mit dem Header
        problematic_lines = []
        
        for i, line in enumerate(lines[1:], 1):  # Überspringe den Header
            original_line = line
            
            # Überprüfe, ob die Zeile die erwartete Anzahl von Feldern hat
            if line.strip():  # Überspringe leere Zeilen
                # Zähle die tatsächlichen Felder (berücksichtige Anführungszeichen)
                in_quotes = False
                field_count = 1  # Beginne mit 1 für das erste Feld
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        field_count += 1
                
                # Behandle unvollständige Zeilen
                if field_count < expected_fields:
                    print(f"Zeile {i+1}: Zu wenige Felder ({field_count}/{expected_fields})")
                    
                    # Füge fehlende Kommas hinzu
                    missing_commas = expected_fields - field_count
                    line = line.rstrip() + ',' * missing_commas + '\n'
                    problematic_lines.append((i+1, "Zu wenige Felder", field_count, expected_fields))
                    
                # Behandle Zeilen mit zu vielen Feldern
                elif field_count > expected_fields:
                    print(f"Zeile {i+1}: Zu viele Felder ({field_count}/{expected_fields})")
                    
                    # Extrahiere nur die ersten expected_fields Felder
                    in_quotes = False
                    field_count = 1
                    new_line = ""
                    
                    for char in line:
                        new_line += char
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            field_count += 1
                            if field_count > expected_fields:
                                break
                    
                    line = new_line.rstrip() + '\n'
                    problematic_lines.append((i+1, "Zu viele Felder", field_count, expected_fields))
                
                # Überprüfe und behebe potentielle Probleme mit unvollständigen Anführungszeichen
                quotes_count = line.count('"')
                if quotes_count % 2 == 1:  # Ungerade Anzahl von Anführungszeichen
                    line = line.rstrip() + '"\n'  # Füge schließendes Anführungszeichen hinzu
                    problematic_lines.append((i+1, "Ungerade Anzahl von Anführungszeichen", quotes_count, quotes_count+1))
            
            # Wenn die Zeile vollständig leer ist, überspringe sie
            if not line.strip():
                problematic_lines.append((i+1, "Leere Zeile", 0, expected_fields))
                continue
                
            fixed_lines.append(line)
        
        # Schreibe die korrigierte Datei
        output_file = file_path_str.replace('.csv', '_fixed.csv')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        # Zeige eine Zusammenfassung der Probleme
        if problematic_lines:
            print(f"\nZusammenfassung der Probleme ({len(problematic_lines)} Zeilen):")
            problem_types = {}
            for _, problem_type, _, _ in problematic_lines:
                problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
                
            for problem_type, count in problem_types.items():
                print(f"  - {problem_type}: {count} Zeilen")
        
        print(f"\nKorrigierte Datei gespeichert als: {output_file}")
        
        # Teste, ob die Datei jetzt gelesen werden kann
        try:
            df = pd.read_csv(output_file, on_bad_lines='warn')
            print(f"✓ CSV-Formatierungsfehler behoben! Datei enthält {len(df)} Zeilen.")
            return output_file
        except Exception as e:
            print(f"⚠️ Korrektur war nicht vollständig erfolgreich: {e}")
            
            # Versuche eine radikalere Korrektur: Extrahiere den Header und erstelle eine minimale gültige CSV
            try:
                print("\nVersuche alternative Reparaturmethode...")
                
                # Extrahiere die Spalten aus dem Header
                columns = header.split(',')
                
                # Erstelle einen neuen DataFrame mit den korrekten Spalten
                import io
                from csv import reader
                
                # Lese die Zeilen als CSV-Datei
                rows = []
                csv_reader = reader(io.StringIO(''.join(fixed_lines)))
                for row in csv_reader:
                    # Stelle sicher, dass jede Zeile die richtige Anzahl von Spalten hat
                    if len(row) == len(columns):
                        rows.append(row)
                
                # Erstelle den DataFrame
                df = pd.DataFrame(rows[1:], columns=columns)  # Überspringe den Header
                
                # Speichere den reparierten DataFrame
                output_file = file_path_str.replace('.csv', '_repaired.csv')
                df.to_csv(output_file, index=False)
                
                print(f"✓ Alternative Reparatur erfolgreich! Datei enthält {len(df)} Zeilen.")
                return output_file
            except Exception as e2:
                print(f"⚠️ Alternative Reparatur fehlgeschlagen: {e2}")
                return output_file
            
    except Exception as e:
        print(f"Fehler beim Versuch, die CSV-Datei zu reparieren: {e}")
        return file_path_str


def repair_csv_structure(file_path, header_only=False):
    """
    Repariert die Struktur einer beschädigten CSV-Datei, die durch falsch behandelte
    Zeilenumbrüche in Feldern verursacht wurde, indem der Inhalt neu strukturiert wird.

    Args:
        file_path (str): Pfad zur zu reparierenden CSV-Datei
        header_only (bool): Wenn True, wird nur der Header verwendet und die Daten neu strukturiert

    Returns:
        str: Pfad zur reparierten Datei
    """
    import csv
    
    print(f"Repariere beschädigte CSV-Struktur in {file_path}...")
    
    # Konvertiere zu string für konsistente Verarbeitung
    file_path_str = str(file_path)
    
    try:
        # Versuch 1: Direktes Laden mit pandas
        try:
            # Aktualisierte Parameter für neuere pandas-Versionen
            df = pd.read_csv(file_path, on_bad_lines='warn', engine='python')
            print(f"DataFrame konnte geladen werden mit {len(df)} Zeilen")
            
            if header_only:
                # Erstelle einen leeren DataFrame mit dem Header
                columns = df.columns
                empty_df = pd.DataFrame(columns=columns)
                
                # Speichere mit sicheren Optionen
                output_file = file_path_str.replace('.csv', '_header_only.csv')
                
                # Speichere mit erhöhter Sicherheit
                empty_df.to_csv(
                    output_file,
                    index=False,
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\',
                    doublequote=True,
                    lineterminator='\n'
                )
                
                print(f"✓ Leere Datei mit Header gespeichert als: {output_file}")
                return output_file
            
            # Speichere mit sicheren Optionen
            output_file = file_path_str.replace('.csv', '_repaired.csv')
            
            # Speichere mit erhöhter Sicherheit
            df.to_csv(
                output_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                doublequote=True,
                lineterminator='\n'
            )
            
            print(f"✓ Reparierte Datei gespeichert als: {output_file}")
            return output_file
        except Exception as e:
            print(f"Fehler beim direkten Laden mit pandas: {e}")
            print("Versuche alternative Methode...")
    
        # Versuch 2: Manuelles Parsen und Neustrukturieren
        # Extrahiere den Header
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Erste Zeile enthält den Header
            header_line = f.readline().strip()
            columns = header_line.split(',')
            expected_fields = len(columns)
            
            print(f"Header hat {expected_fields} Felder: {columns}")
            
            if header_only:
                # Erstelle einen leeren DataFrame mit dem Header
                empty_df = pd.DataFrame(columns=columns)
                output_file = file_path_str.replace('.csv', '_header_only.csv')
                
                # Speichere mit erhöhter Sicherheit
                empty_df.to_csv(
                    output_file,
                    index=False,
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\',
                    doublequote=True,
                    lineterminator='\n'
                )
                
                print(f"✓ Leere Datei mit Header gespeichert als: {output_file}")
                return output_file
        
        # Versuch 3: CSV-Reader mit maximaler Flexibilität
        print("Versuche, die Daten mit CSV-Reader zu parsen...")
        
        # Erhöhe das Feldlimit für große Felder
        import sys
        csv.field_size_limit(sys.maxsize)
        
        # Verwende einen Unicode-Reader mit Fehlerbehandlung
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Überpspringe den Header, da wir ihn bereits haben
            next(f)
            
            # Lese den Rest der Datei
            content = f.read()
        
        # Liste für gültige Datensätze
        valid_records = []
        
        # Versuche, die Daten aus dem originalen CSV zu extrahieren
        # indem wir ein temporäres CSV erstellen und den CSV-Reader verwenden
        import io
        
        # Erstelle einen flexiblen CSV-Reader
        csv_reader = csv.reader(
            io.StringIO(content),
            quotechar='"',
            delimiter=',',
            quoting=csv.QUOTE_MINIMAL,
            skipinitialspace=True
        )
        
        # Sammle gültige Datensätze
        success_count = 0
        error_count = 0
        for i, row in enumerate(csv_reader):
            if len(row) == expected_fields:
                valid_records.append(row)
                success_count += 1
                if success_count % 1000 == 0:
                    print(f"Gefunden: {success_count} gültige Datensätze")
            else:
                error_count += 1
                if error_count < 10:  # Begrenze die Anzahl der angezeigten Fehler
                    print(f"Überspringe Zeile mit {len(row)} statt {expected_fields} Feldern")
        
        print(f"Extrahiert: {len(valid_records)} gültige Datensätze")
        print(f"Fehlerhafte Zeilen: {error_count}")
        
        # Erstelle einen neuen DataFrame
        repaired_df = pd.DataFrame(valid_records, columns=columns)
        
        # Speichere den reparierten DataFrame
        output_file = file_path_str.replace('.csv', '_repaired.csv')
        
        # Speichere mit erhöhter Sicherheit
        repaired_df.to_csv(
            output_file,
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            doublequote=True,
            lineterminator='\n'
        )
        
        print(f"✓ Reparierte Datei gespeichert als: {output_file} mit {len(repaired_df)} Datensätzen")
        return output_file
        
    except Exception as e:
        print(f"Fehler bei der Reparatur der CSV-Datei: {e}")
        
        # Notfallplan: Leere Datei mit Header erstellen
        try:
            print("Notfall-Reparatur: Erstelle leere Datei mit Header...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header_line = f.readline().strip()
                
            output_file = file_path_str.replace('.csv', '_empty.csv')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(header_line + '\n')
                
            print(f"✓ Leere Datei mit Header gespeichert als: {output_file}")
            return output_file
        except Exception as e2:
            print(f"Auch Notfall-Reparatur fehlgeschlagen: {e2}")
            return file_path_str 