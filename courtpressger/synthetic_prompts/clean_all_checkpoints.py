#!/usr/bin/env python3
import os
import sys
import glob
from clean_checkpoints import clean_checkpoint_file

def clean_all_checkpoints(directory="checkpoints"):
    # Finde alle CSV-Dateien im Checkpoint-Verzeichnis
    checkpoint_files = glob.glob(os.path.join(directory, "synthetic_prompts_*.csv"))
    
    # Filtere bereits bereinigte Dateien heraus
    checkpoint_files = [f for f in checkpoint_files if not (f.endswith("_clean.csv") or f.endswith("_original.csv"))]
    
    print(f"Gefundene Checkpoint-Dateien: {len(checkpoint_files)}")
    
    for file_path in checkpoint_files:
        print(f"\nVerarbeite: {file_path}")
        cleaned_file = clean_checkpoint_file(file_path)
        print(f"Bereinigung abgeschlossen: {cleaned_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "checkpoints"
    
    clean_all_checkpoints(directory)
    print("\nAlle Checkpoint-Dateien wurden bereinigt.") 