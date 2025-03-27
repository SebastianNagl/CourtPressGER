"""
Generator module for creating synthetic prompts from court rulings and press releases.

This module provides the core functionality to generate synthetic prompts
using the Anthropic Claude API.
"""

import os
import time
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import anthropic

from .rate_limiter import RateLimiter, estimate_token_count

# Erstelle eine globale Rate Limiter-Instanz
rate_limiter = RateLimiter()


def generate_synthetic_prompt(court_ruling, press_release, client=None, model="claude-3-7-sonnet-20250219", retries=3, wait_time=2):
    """
    Generiert einen synthetischen Prompt, der die gegebene Pressemitteilung aus dem Gerichtsurteil erzeugen könnte.
    
    Args:
        court_ruling (str): Der Text des Gerichtsurteils
        press_release (str): Der Text der Pressemitteilung
        client (anthropic.Anthropic, optional): Ein existierender Claude API-Client
        model (str): Das zu verwendende Claude-Modell
        retries (int): Anzahl der Wiederholungsversuche bei API-Fehlern
        wait_time (int): Wartezeit zwischen Wiederholungsversuchen in Sekunden
        
    Returns:
        str: Der generierte synthetische Prompt
    """
    # Initialisiere den Claude-Client, falls nicht übergeben
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY ist nicht in der Umgebung gesetzt")
        client = anthropic.Anthropic(api_key=api_key)
    
    # Vorbereitung des System-Prompts
    system_prompt = """
    Du bist ein Experte für juristische Texte und Kommunikation. Deine Aufgabe ist es, ein Gerichtsurteil und die 
    dazugehörige Pressemitteilung zu analysieren und dann herauszufinden, welcher Prompt verwendet worden sein könnte, 
    um diese Pressemitteilung aus dem Gerichtsurteil zu generieren, wenn man ihn einem LLM gegeben hätte.
    
    1. Analysiere, wie die Pressemitteilung Informationen aus dem Urteil vereinfacht, umstrukturiert und Schlüsselinformationen hervorhebt
    2. Berücksichtige den Ton, die Struktur und den Detaillierungsgrad der Pressemitteilung
    3. Identifiziere, welche Anweisungen nötig wären, um den juristischen Text in diese Pressemitteilung zu transformieren
    
    Erkläre NICHT deine Überlegungen und füge KEINE Meta-Kommentare hinzu. Gib NUR den tatsächlichen Prompt aus, der die 
    Pressemitteilung aus dem Gerichtsurteil generieren würde. Sei spezifisch und detailliert in deinem synthetisierten Prompt.
    """
    
    # Vorbereitung des Benutzer-Prompts
    user_prompt = f"""
    Hier ist das originale Gerichtsurteil:
    
    ```
    {court_ruling}
    ```
    
    Und hier ist die Pressemitteilung, die daraus erstellt wurde:
    
    ```
    {press_release}
    ```
    
    Erstelle einen detaillierten Prompt, der einem LLM gegeben werden könnte, um die obige Pressemitteilung aus dem Gerichtsurteil zu generieren. 
    Schreibe NUR den Prompt selbst, ohne Erklärungen oder Meta-Kommentare.
    """
    
    # Schätze den Token-Verbrauch
    input_tokens_estimate = estimate_token_count(system_prompt) + estimate_token_count(user_prompt)
    output_tokens_estimate = 500  # Konservative Schätzung für die Antwortlänge
    
    # Warte, wenn nötig, um Rate-Limits einzuhalten
    rate_limiter.wait_if_needed(input_tokens_estimate, output_tokens_estimate)
    
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Zeichne tatsächliche Token-Nutzung auf, falls verfügbar
            input_tokens = response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else input_tokens_estimate
            output_tokens = response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else output_tokens_estimate
            rate_limiter.record_usage(input_tokens, output_tokens)
            
            return response.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Fehler: {e}. Neuer Versuch in {wait_time} Sekunden...")
                time.sleep(wait_time)
            else:
                print(f"Nach {retries} Versuchen fehlgeschlagen: {e}")
                return f"Fehler bei der Generierung des Prompts: {e}"


def process_batch(df, batch_size=5, start_idx=0, save_interval=10, fix_errors=False, checkpoint_dir=None, output_prefix=None, client=None):
    """
    Verarbeitet einen Dataframe in Batches und generiert synthetische Prompts für jede Zeile.
    
    Args:
        df (pd.DataFrame): Der Dataframe mit Gerichtsurteilen und Pressemitteilungen
        batch_size (int): Anzahl der Elemente, die vor dem Speichern von Zwischenergebnissen verarbeitet werden
        start_idx (int): Index, bei dem die Verarbeitung beginnen soll (für die Wiederaufnahme)
        save_interval (int): Wie oft Zwischenergebnisse gespeichert werden sollen
        fix_errors (bool): Wenn True, werden Zeilen mit API-Fehlermeldungen erneut verarbeitet
        checkpoint_dir (Path, optional): Verzeichnis für Checkpoints, standardmäßig 'checkpoints' im aktuellen Verzeichnis
        output_prefix (str, optional): Präfix für den Dateinamen der Checkpoints
        client (anthropic.Anthropic, optional): Ein existierender Claude API-Client
        
    Returns:
        pd.DataFrame: Der Dataframe mit hinzugefügten synthetischen Prompts
    """
    # Initialisiere den Claude-Client, falls nicht übergeben
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY ist nicht in der Umgebung gesetzt")
        client = anthropic.Anthropic(api_key=api_key)
    
    # Erstelle eine synthetic_prompt-Spalte, falls sie nicht existiert
    if 'synthetic_prompt' not in df.columns:
        df['synthetic_prompt'] = None
    
    # Identifiziere Zeilen mit API-Fehlermeldungen, falls fix_errors=True
    if fix_errors:
        error_mask = df['synthetic_prompt'].astype(str).str.contains("Fehler bei der Generierung des Prompts", na=False)
        error_indices = df[error_mask].index.tolist()
        if error_indices:
            print(f"Gefunden: {len(error_indices)} Einträge mit API-Fehlern")
            # Setze die fehlerhaften Einträge auf None zurück, damit sie neu verarbeitet werden
            df.loc[error_indices, 'synthetic_prompt'] = None
    
    # Erstelle ein Verzeichnis für Checkpoints, falls es nicht existiert
    if checkpoint_dir is None:
        checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Standard-Präfix für Checkpoints
    if output_prefix is None:
        output_prefix = "synthetic_prompts_checkpoint"
    
    # Verarbeite in Batches
    for i in tqdm(range(start_idx, len(df), batch_size)):
        batch_end = min(i + batch_size, len(df))
        batch = df.iloc[i:batch_end].copy()
        
        # Zähler für erfolgreiche Verarbeitungen in diesem Batch
        successful_in_batch = 0
        
        for idx, row in batch.iterrows():
            if pd.isna(df.at[idx, 'synthetic_prompt']):
                court_ruling = row['judgement']
                press_release = row['summary']
                
                # Kürze sehr lange Eingaben, um Token-Limits zu vermeiden
                if len(court_ruling) > 12000:  # Reduziert für bessere Tokenbegrenzung
                    court_ruling = court_ruling[:12000] + "..."
                if len(press_release) > 4000:  # Reduziert für bessere Tokenbegrenzung
                    press_release = press_release[:4000] + "..."
                
                # Generiere den synthetischen Prompt
                synthetic_prompt = generate_synthetic_prompt(court_ruling, press_release, client=client)
                
                # Prüfe, ob die Antwort einen API-Fehler enthält
                if "Fehler bei der Generierung des Prompts" in str(synthetic_prompt):
                    print(f"⚠️ API-Fehler bei Index {idx}, wird in einem zukünftigen Durchlauf erneut versucht")
                else:
                    successful_in_batch += 1
                
                df.at[idx, 'synthetic_prompt'] = synthetic_prompt
                
                # Zeige Fortschritt an
                if (idx - i) % 5 == 0 or idx == batch_end - 1:
                    print(f"{idx+1}/{len(df)} Einträge verarbeitet - {successful_in_batch}/{batch_end-i} erfolgreich in diesem Batch")
        
        # Speichere Checkpoint in regelmäßigen Abständen
        if (i // batch_size) % save_interval == 0 or batch_end == len(df):
            checkpoint_path = checkpoint_dir / f"{output_prefix}_{batch_end}.csv"
            df.to_csv(checkpoint_path, index=False)
            print(f"Checkpoint gespeichert unter {checkpoint_path}")
            
            # Kurze Pause nach jedem Batch, um Rate-Limits zu entspannen
            time.sleep(1)
    
    return df