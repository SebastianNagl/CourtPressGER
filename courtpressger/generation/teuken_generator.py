#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul zur Generierung von Pressemitteilungen mit dem Teuken-Modell.
"""

import os
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_with_teuken(
    dataset_path: str,
    output_path: str,
    model_path: str = "models/teuken",
    ruling_column: str = "judgement",
    prompt_column: str = "synthetic_prompt",
    output_column: str = "generated_pr_teuken",
    batch_size: int = 10,
    max_length: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.95
):
    """
    Generiert Pressemitteilungen mit dem Teuken-Modell und speichert die Ergebnisse.
    
    Args:
        dataset_path: Pfad zur Eingabedatei (CSV)
        output_path: Pfad zur Ausgabedatei (CSV)
        model_path: Pfad zum Teuken-Modell
        ruling_column: Name der Spalte mit Gerichtsurteilen
        prompt_column: Name der Spalte mit synthetischen Prompts
        output_column: Name der Spalte für die generierten Pressemitteilungen
        batch_size: Anzahl der Einträge für Checkpoint-Speicherung
        max_length: Maximale Länge der generierten Texte
        temperature: Temperatur für die Generierung (0-1)
        do_sample: Sampling aktivieren
        top_p: Top-p für Nucleus-Sampling
    """
    print(f"Lade Datensatz aus {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Datensatz geladen: {len(df)} Einträge")

    print(f"Lade Teuken-Modell aus {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("Modell geladen")

    # Generierungsparameter
    gen_config = {
        "max_length": max_length,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p
    }

    # Liste für die generierten Pressemitteilungen
    generated_texts = []

    # Erstelle Ausgabeverzeichnis, falls nicht vorhanden
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Generierung für jeden Eintrag im Datensatz
    print("Starte Generierung...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ruling = row[ruling_column]
        prompt = row[prompt_column]
        
        # Prompt-Formatierung
        combined_prompt = f"{prompt}\n\nGerichtsurteil:\n{ruling}"
        
        # Tokenisierung und Generierung
        inputs = tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Entferne token_type_ids, da das Teuken-Modell diesen Parameter nicht unterstützt
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        
        # Verschiebe Eingaben auf das Gerät des Modells
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generierung
        outputs = model.generate(
            **inputs,
            **gen_config
        )
        
        # Dekodierung
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Speichern der generierten Pressemitteilung
        generated_texts.append(generated_text)
        
        # Checkpoint-Speicherung alle batch_size Einträge
        if (idx + 1) % batch_size == 0:
            print(f"Checkpoint: {idx + 1}/{len(df)} Einträge verarbeitet")
            temp_df = df.copy()
            temp_df[output_column] = generated_texts + [""] * (len(df) - len(generated_texts))
            temp_df.to_csv(output_path, index=False)

    # Füge die generierten Pressemitteilungen zum Datensatz hinzu
    df[output_column] = generated_texts
    
    # Speichere den aktualisierten Datensatz
    print(f"Speichere Ergebnisse in {output_path}...")
    df.to_csv(output_path, index=False)
    print("Fertig!")
    
    return df

if __name__ == "__main__":
    # Default-Einstellungen, wenn direkt ausgeführt
    dataset_path = "data/processed/cases_prs_synth_prompts_subset.csv"
    output_path = "data/processed/cases_prs_synth_prompts_subset_with_teuken.csv"
    
    generate_with_teuken(
        dataset_path=dataset_path,
        output_path=output_path
    ) 