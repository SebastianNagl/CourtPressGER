#!/usr/bin/env python3
import pandas as pd
import re
import os
import sys

def clean_checkpoint_file(file_path):
    print(f"Bereinige Datei: {file_path}")
    
    # Lese die CSV-Datei
    try:
        df = pd.read_csv(file_path, on_bad_lines='warn')
        print(f"Original enthält {len(df)} Zeilen")
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Datei: {e}")
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
        output_file = file_path.replace('.csv', '_clean.csv')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        print(f"Bereinigte Datei gespeichert als: {output_file}")
        return output_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "checkpoints/synthetic_prompts_2130.csv"
    
    cleaned_file = clean_checkpoint_file(file_path)
    print(f"Bereinigung abgeschlossen. Ergebnis: {cleaned_file}") 