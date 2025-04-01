#!/usr/bin/env python3
"""
Skript zum Herunterladen und Speichern des EuroBERT-Modells.
"""
import os
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer

def main():
    # Modell-Pfad
    model_path = Path("models/eurobert")
    model_path.mkdir(parents=True, exist_ok=True)
    
    print("Lade EuroBERT-Modell herunter...")
    model = AutoModelForMaskedLM.from_pretrained(
        "EuroBERT/EuroBERT-2.1B",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EuroBERT/EuroBERT-2.1B",
        trust_remote_code=True
    )
    
    print(f"Speichere Modell in {model_path}...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print("Fertig!")

if __name__ == "__main__":
    main() 