"""
Hilfsfunktionen für die Datenbereinigung, einschließlich Text-Vorverarbeitung
und GPU-Ressourcenmanagement.
"""

import re
import warnings
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any, Callable

# Spacy für NLP-Verarbeitung
import spacy

# GPU-Beschleunigung (optional)
GPU_AVAILABLE = False
try:
    import cudf
    import cuml
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.decomposition import PCA as cuPCA
    from cuml.cluster import KMeans as cuKMeans
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.metrics import pairwise_distances
    from cuml.preprocessing import normalize

    # GPU-Speicherverwaltung (optional)
    try:
        import rmm
        rmm.reinitialize(managed_memory=True)
        pool = rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())
        rmm.mr.set_current_device_resource(pool)
        print("RMM memory pool initialisiert für optimierte GPU-Speicherverwaltung")
    except ImportError:
        print("RMM nicht verfügbar, verwende Standard-Speicherverwaltung")

    GPU_AVAILABLE = True
    print("GPU-Beschleunigung aktiviert (RAPIDS-Bibliotheken geladen)")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU-Beschleunigung nicht verfügbar, verwende CPU-Version")

# Warnungen unterdrücken
warnings.filterwarnings('ignore')


def load_spacy_model(model_name: str = "de_core_news_sm") -> Any:
    """
    Lädt ein Spacy-Modell und versucht, GPU-Beschleunigung zu verwenden, wenn verfügbar.

    Args:
        model_name: Name des zu ladenden Spacy-Modells

    Returns:
        Das geladene Spacy-Modell
    """
    try:
        # Prüfen, ob GPU-beschleunigtes Spacy verfügbar ist
        if GPU_AVAILABLE:
            try:
                import spacy_cuda
                nlp = spacy_cuda.load(model_name)
                print(f"SpaCy {model_name} mit GPU-Beschleunigung geladen")
                return nlp
            except (ImportError, ModuleNotFoundError):
                # Fallback auf CPU-Version
                nlp = spacy.load(model_name)
                print(f"SpaCy {model_name} (CPU-Version) geladen")
                return nlp
        else:
            # Standard-CPU-Version
            nlp = spacy.load(model_name)
            print(f"SpaCy {model_name} geladen")
            return nlp
    except (OSError, IOError) as e:
        print(f"SpaCy {model_name} wird installiert...")
        import subprocess
        subprocess.run([f"python -m spacy download {model_name}"], shell=True)
        nlp = spacy.load(model_name)
        print(f"SpaCy {model_name} geladen")
        return nlp


def preprocess_text(text: str, nlp: Any, batch_size: int = 1000) -> str:
    """
    GPU-optimierte Version der Textvorverarbeitung wenn möglich, sonst CPU-Fallback.

    Args:
        text: Zu verarbeitender Text
        nlp: Geladenes Spacy-Modell
        batch_size: Größe der Textblöcke für die Verarbeitung

    Returns:
        Vorverarbeiteter Text mit lemmatisierten Tokens
    """
    if pd.isna(text):
        return ""

    text = str(text)

    if GPU_AVAILABLE and len(text) > 5000:
        # Für lange Texte: Chunking für bessere GPU-Nutzung
        chunks = [text[i:i+batch_size]
                  for i in range(0, len(text), batch_size)]
        processed_chunks = []

        for chunk in chunks:
            doc = nlp(chunk)
            # Nur Substantive, Verben, Adjektive und Adverbien behalten
            tokens = [token.lemma_.lower() for token in doc
                      if not token.is_stop and not token.is_punct
                      and token.is_alpha and len(token.text) > 2
                      and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
            processed_chunks.append(" ".join(tokens))

        return " ".join(processed_chunks)
    else:
        # Standard-Verarbeitung für kurze Texte oder CPU-Modus
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct
                  and token.is_alpha and len(token.text) > 2
                  and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        return " ".join(tokens)


def batch_preprocess_texts(texts: pd.Series, nlp: Any, batch_size: int = 32) -> pd.Series:
    """
    Batch-Verarbeitung für bessere GPU-Auslastung bei der Textvorverarbeitung.

    Args:
        texts: Serie mit zu verarbeitenden Texten
        nlp: Geladenes Spacy-Modell
        batch_size: Anzahl der Texte pro Batch

    Returns:
        Serie mit vorverarbeiteten Texten
    """
    if GPU_AVAILABLE:
        # Für GPU: Batch-Verarbeitung
        result = []
        texts = texts.fillna("").tolist()

        for i in tqdm(range(0, len(texts), batch_size), desc="Vorverarbeitung"):
            batch = texts[i:i+batch_size]
            processed_batch = [preprocess_text(text, nlp) for text in batch]
            result.extend(processed_batch)

            # Explizite Speicherfreigabe für GPU
            if i % (batch_size * 10) == 0 and 'cp' in globals():
                cp.get_default_memory_pool().free_all_blocks()

        return pd.Series(result, index=range(len(result)))
    else:
        # Für CPU: Einzelverarbeitung
        return texts.fillna("").apply(lambda text: preprocess_text(text, nlp))


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Lädt den Datensatz und führt grundlegende Validierung durch.
    Unterstützt sowohl einzelne CSV-Dateien als auch das neue Format mit cleaned/removed.

    Args:
        file_path: Pfad zur CSV-Datei oder zum Verzeichnis mit cleaned/removed

    Returns:
        Geladener DataFrame
    """
    try:
        if Path(file_path).is_dir():
            # Neues Format: Verzeichnis mit cleaned/removed
            cleaned_path = Path(file_path) / "cleaned.csv"
            removed_path = Path(file_path) / "removed.csv"
            
            if not cleaned_path.exists() or not removed_path.exists():
                raise FileNotFoundError("Verzeichnis muss cleaned.csv und removed.csv enthalten")
            
            df_cleaned = pd.read_csv(cleaned_path)
            df_removed = pd.read_csv(removed_path)
            
            # Füge Status-Spalte hinzu
            df_cleaned['status'] = 'cleaned'
            df_removed['status'] = 'removed'
            
            # Kombiniere die DataFrames
            df = pd.concat([df_cleaned, df_removed], ignore_index=True)
            print(f"Daten geladen: {len(df)} Einträge ({len(df_cleaned)} bereinigt, {len(df_removed)} entfernt)")
            return df
        else:
            # Altes Format: Einzelne CSV-Datei
            df = pd.read_csv(file_path)
            print(f"Daten geladen: {len(df)} Einträge")
            return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        raise


def save_results(df: pd.DataFrame, output_dir: str, filename: str = None) -> None:
    """
    Speichert die Ergebnisse im neuen Format (cleaned/removed) oder als einzelne Datei.

    Args:
        df: DataFrame mit den Ergebnissen
        output_dir: Ausgabeverzeichnis
        filename: Optionaler Dateiname für das alte Format
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if 'status' in df.columns:
        # Neues Format: Speichere als cleaned/removed
        df_cleaned = df[df['status'] == 'cleaned'].copy()
        df_removed = df[df['status'] == 'removed'].copy()
        
        # Entferne Status-Spalte
        df_cleaned = df_cleaned.drop('status', axis=1)
        df_removed = df_removed.drop('status', axis=1)
        
        # Speichere getrennte Dateien
        df_cleaned.to_csv(output_path / "cleaned.csv", index=False)
        df_removed.to_csv(output_path / "removed.csv", index=False)
        print(f"Ergebnisse gespeichert in {output_path}")
    else:
        # Altes Format: Speichere als einzelne Datei
        if filename is None:
            filename = "results.csv"
        df.to_csv(output_path / filename, index=False)
        print(f"Ergebnisse gespeichert in {output_path / filename}")
