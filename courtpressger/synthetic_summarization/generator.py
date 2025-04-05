import os
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import anthropic
import glob

from courtpressger.synthetic_prompts.rate_limiter import RateLimiter, estimate_token_count
from courtpressger.synthetic_prompts.sanitizer import sanitize_api_response

rate_limiter = RateLimiter()

def generate_synthetic_summary(court_ruling, client=None, model="claude-3-7-sonnet-20250219", retries=3, wait_time=2):
    # Initialisiere den Claude-Client, falls nicht übergeben
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY ist nicht in der Umgebung gesetzt")
        client = anthropic.Anthropic(api_key=api_key)
    
    # Vorbereitung des System-Prompts
    system_prompt = """
    Du bist ein Experte für juristische Texte. Deine Aufgabe ist es, ein Gerichtsurteil präzise zu kondensieren.

    Befolge strikt diese Richtlinien:
    1. Kürze den Text auf MAXIMAL 800 Tokens (etwa 600 Wörter)
    2. Behalte die originale Struktur und wichtigsten Gliederungspunkte bei
    3. Behalte die wesentliche juristische Fachsprache bei
    4. Fokussiere dich auf:
       - Aktenzeichen und Hauptparteien
       - Entscheidungsformel (Tenor)
       - Kernargumente des Gerichts
       - Wesentliche Rechtsgrundlagen
    5. Fasse jeden Abschnitt stark zusammen, behalte aber alle wesentlichen Strukturelemente
    6. Achte darauf, dass die Zusammenfassung VOLLSTÄNDIG ist und nicht mitten im Gedankengang abbricht

    Deine Zusammenfassung MUSS unter 1000 Tokens liegen und trotzdem vollständig sein.
    """

    # Benutzer-Prompt mit dem Gerichtsurteil
    user_prompt = f"""
    Kondensiere das folgende Gerichtsurteil auf MAXIMAL 1000 Tokens (ca. 750 Wörter). Behalte nur die wesentlichsten juristischen Inhalte und Kernargumente. Die Zusammenfassung MUSS vollständig sein, mit allen wichtigen Teilen des Urteils.

    Hier ist das Gerichtsurteil:

    ```
    {court_ruling}
    ```
    WICHTIG: Deine Antwort darf 1000 Tokens nicht überschreiten und muss trotzdem einen vollständigen Überblick über das Urteil geben.
    """
    input_tokens_estimate = estimate_token_count(system_prompt) + estimate_token_count(user_prompt)
    output_tokens_estimate = 1000  # Konservative Schätzung für die Antwortlänge
    
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
            
            # Extrahiere und bereinige die Antwort
            raw_response = response.content[0].text.strip()
            sanitized_response = sanitize_api_response(raw_response)
            
            return sanitized_response
        except Exception as e:
            if attempt < retries - 1:
                print(f"Fehler: {e}. Neuer Versuch in {wait_time} Sekunden...")
                time.sleep(wait_time)
            else:
                print(f"Nach {retries} Versuchen fehlgeschlagen: {e}")
                return f"Fehler bei der Generierung des Prompts: {e}"

def process_batch(df, batch_size=10, client=None):
    
    # Initialisiere den Claude-Client, falls nicht übergeben
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY ist nicht in der Umgebung gesetzt")
        client = anthropic.Anthropic(api_key=api_key)

    if 'synthetic_summary' not in df.columns:
        df['synthetic_summary'] = None

    start_idx = 0

    for i in tqdm(range(start_idx, len(df), batch_size)):
        batch_end = min(i + batch_size, len(df))
        batch = df.iloc[i:batch_end].copy()
                
        # Zähler für erfolgreiche Verarbeitungen in diesem Batch
        successful_in_batch = 0

        for idx, row in batch.iterrows():
            if pd.isna(df.at[idx, 'synthetic_summary']):
                court_ruling = row['judgement']
            
                if len(court_ruling) > 12000:  # Reduziert für bessere Tokenbegrenzung
                        court_ruling = court_ruling[:12000] + "..."
                synthetic_summary = generate_synthetic_summary(court_ruling, client=client)
                # Prüfe, ob die Antwort einen API-Fehler enthält
                if "Fehler bei der Generierung des Prompts" in str(synthetic_summary):
                    print(f"⚠️ API-Fehler bei Index {idx}, wird in einem zukünftigen Durchlauf erneut versucht")
                else:
                    successful_in_batch += 1
                
                df.at[idx, 'synthetic_summary'] = synthetic_summary
                
                # Zeige Fortschritt an
                if (idx - i) % 5 == 0 or idx == batch_end - 1:
                    print(f"{idx+1}/{len(df)} Einträge verarbeitet - {successful_in_batch}/{batch_end-i} erfolgreich in diesem Batch")
    return df


