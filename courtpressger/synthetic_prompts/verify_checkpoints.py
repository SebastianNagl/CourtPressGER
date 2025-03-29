#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd

def verify_checkpoint_file(file_path):
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
    # Finde alle bereinigten CSV-Dateien
    clean_files = glob.glob(os.path.join(directory, "*_clean.csv"))
    
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            verify_all_checkpoints(sys.argv[1])
        else:
            verify_checkpoint_file(sys.argv[1])
    else:
        verify_all_checkpoints("checkpoints") 