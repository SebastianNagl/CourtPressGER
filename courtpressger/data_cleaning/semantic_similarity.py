"""
Semantische Ähnlichkeitsanalyse zwischen Urteilen und Pressemitteilungen.

Berechnet die semantische Ähnlichkeit zwischen Urteilen und Pressemitteilungen
zur Identifikation von irrelevanten oder nicht zusammengehörigen Paaren.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from pathlib import Path
from typing import Tuple, Optional

from courtpressger.data_cleaning.utils import (
    load_dataset, save_results, load_spacy_model,
    batch_preprocess_texts, GPU_AVAILABLE
)

# Importiere GPU-beschleunigte Versionen falls verfügbar
if GPU_AVAILABLE:
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.preprocessing import normalize


def compute_tfidf_similarity(
    df: pd.DataFrame,
    max_features: int = 10000,
    min_df: int = 5,
    max_df: float = 0.8,
    ngram_range: Tuple[int, int] = (1, 2),
    preprocess: bool = True,
    save_visualizations: bool = False,
    output_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Berechnet die TF-IDF-basierte Kosinus-Ähnlichkeit zwischen Urteilen und Pressemitteilungen.

    Args:
        df: DataFrame mit Urteilen und Pressemitteilungen
        max_features: Maximale Anzahl an Features für TF-IDF
        min_df: Minimale Dokument-Frequenz für TF-IDF
        max_df: Maximale Dokument-Frequenz für TF-IDF
        ngram_range: N-Gramm-Bereich für TF-IDF
        preprocess: Ob Texte vorverarbeitet werden sollen
        save_visualizations: Ob Visualisierungen gespeichert werden sollen
        output_dir: Ausgabeverzeichnis

    Returns:
        DataFrame mit berechneten Ähnlichkeitswerten
    """
    print("\n## Semantische Ähnlichkeit mit TF-IDF Embeddings")

    # Initialisiere TF-IDF Vektorisierer
    print("Berechne TF-IDF Vektoren...")

    if GPU_AVAILABLE:
        tfidf_vectorizer = cuTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
    else:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=True
        )

    # Führe Vorverarbeitung durch, wenn gewünscht
    if preprocess:
        print("Führe Text-Vorverarbeitung durch...")
        # Lade Spacy Modell
        nlp = load_spacy_model()
        # Verarbeite Texte
        df['preprocessed_summary'] = batch_preprocess_texts(df['summary'], nlp)
        df['preprocessed_judgement'] = batch_preprocess_texts(
            df['judgement'], nlp)

        summary_texts = df['preprocessed_summary']
        judgement_texts = df['preprocessed_judgement']
    else:
        # Verwende Rohtexte
        summary_texts = df['summary'].fillna('')
        judgement_texts = df['judgement'].fillna('')

    # Kombiniere alle Texte für das Training des Vektorisierers
    all_texts = list(summary_texts) + list(judgement_texts)
    tfidf_vectorizer.fit(all_texts)
    print(f"Vokabulargröße: {len(tfidf_vectorizer.get_feature_names_out())}")

    # Vektorisiere Urteile und Pressemitteilungen
    summary_vectors = tfidf_vectorizer.transform(summary_texts)
    judgement_vectors = tfidf_vectorizer.transform(judgement_texts)

    # Berechne die Kosinus-Ähnlichkeit für jedes Paar
    print("Berechne Kosinus-Ähnlichkeiten...")
    similarities = []

    # GPU-beschleunigte oder CPU-basierte Berechnung je nach Verfügbarkeit
    if GPU_AVAILABLE:
        for i in range(len(df)):
            if i < summary_vectors.shape[0] and i < judgement_vectors.shape[0]:
                # Für GPU: normalisierte Vektoren und paarweise Ähnlichkeiten mit cuML
                summary_vec = cp.sparse.csr_matrix(summary_vectors[i])
                judgement_vec = cp.sparse.csr_matrix(judgement_vectors[i])

                # Umwandlung in dichte Arrays (notwendig für einige cuML-Operationen)
                summary_dense = summary_vec.toarray().astype(cp.float32)
                judgement_dense = judgement_vec.toarray().astype(cp.float32)

                # Normalisierung (für Kosinus-Ähnlichkeit)
                if cp.sum(summary_dense) > 0:
                    summary_dense = normalize(summary_dense)
                if cp.sum(judgement_dense) > 0:
                    judgement_dense = normalize(judgement_dense)

                # Kosinus-Ähnlichkeit = Skalarprodukt der normalisierten Vektoren
                similarity = cp.dot(summary_dense.flatten(),
                                    judgement_dense.flatten())
                similarities.append(float(similarity))
            else:
                similarities.append(0.0)

            # Gib alle 1000 Einträge den Fortschritt aus
            if i % 1000 == 0:
                print(f"Fortschritt: {i}/{len(df)} ({i/len(df)*100:.1f}%)")

            # Speicherbereinigung für GPU
            if i % 5000 == 0 and i > 0:
                cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU-Implementierung
        for i in range(len(df)):
            if i < summary_vectors.shape[0] and i < judgement_vectors.shape[0]:
                # Extrahiere die Vektoren für das aktuelle Paar
                summary_vec = summary_vectors[i]
                judgement_vec = judgement_vectors[i]

                # Berechne die Kosinus-Ähnlichkeit
                similarity = cosine_similarity(
                    summary_vec, judgement_vec)[0][0]
                similarities.append(similarity)
            else:
                similarities.append(0.0)

            # Gib alle 1000 Einträge den Fortschritt aus
            if i % 1000 == 0:
                print(f"Fortschritt: {i}/{len(df)} ({i/len(df)*100:.1f}%)")

    # Speichere die Ähnlichkeiten im DataFrame
    df['tfidf_similarity'] = similarities

    # Definiere einen Schwellenwert für die Ähnlichkeit (10% Quantil)
    similarity_threshold = df['tfidf_similarity'].quantile(0.1)
    print(f"Schwellenwert für TF-IDF Ähnlichkeit: {similarity_threshold:.3f}")

    # Identifiziere Paare mit niedriger Ähnlichkeit
    df['is_dissimilar_tfidf'] = df['tfidf_similarity'] < similarity_threshold

    # Auswertung der semantischen Ähnlichkeitsmethode
    dissimilar_count = df['is_dissimilar_tfidf'].sum()
    print(
        f"TF-IDF Ähnlichkeitserkennung: {dissimilar_count} von {len(df)} Einträgen ({dissimilar_count/len(df)*100:.2f}%) haben geringe Ähnlichkeit.")

    # Erstelle Visualisierungen wenn gewünscht
    if save_visualizations:
        create_similarity_visualizations(df, output_dir, similarity_threshold)

    return df


def create_similarity_visualizations(df: pd.DataFrame, output_dir: str, threshold: float) -> None:
    """
    Erstellt Visualisierungen für die semantische Ähnlichkeitsanalyse.

    Args:
        df: DataFrame mit berechneten Ähnlichkeitswerten
        output_dir: Ausgabeverzeichnis für Visualisierungen
        threshold: Schwellenwert für Ähnlichkeit
    """
    output_path = Path(output_dir)
    
    # Überprüfe, ob der Ausgabepfad "reports" enthält
    if "reports" not in str(output_path):
        # Wenn nicht, erstelle einen reports/similarity Unterordner
        parent_dir = output_path.parent if output_path.is_file() else output_path
        output_path = parent_dir / "reports" / "similarity"
    
    output_path.mkdir(exist_ok=True, parents=True)

    # Histogramm der Ähnlichkeitsverteilung
    plt.figure(figsize=(10, 6))
    plt.hist(df['tfidf_similarity'], bins=50)
    plt.title('Verteilung der TF-IDF Kosinus-Ähnlichkeiten')
    plt.xlabel('Kosinus-Ähnlichkeit')
    plt.ylabel('Anzahl der Paare')
    plt.axvline(x=threshold, color='r', linestyle='--',
                label=f'10% Quantil: {threshold:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "tfidf_similarity_distribution.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # Scatterplot der Ähnlichkeiten (falls andere Metriken verfügbar sind)
    if 'is_announcement_rule' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df[~df['is_announcement_rule']]['tfidf_similarity'],
            range(len(df[~df['is_announcement_rule']])),
            alpha=0.5, label='Andere Mitteilungen', s=5
        )
        plt.scatter(
            df[df['is_announcement_rule']]['tfidf_similarity'],
            range(len(df[df['is_announcement_rule']])),
            alpha=0.5, color='red', label='Regelbasierte Ankündigungen', s=5
        )
        plt.axvline(x=threshold, color='black', linestyle='--',
                    label=f'Schwellenwert: {threshold:.3f}')
        plt.title('Semantische Ähnlichkeit nach regelbasierter Klassifikation')
        plt.xlabel('Kosinus-Ähnlichkeit')
        plt.ylabel('Datensatz-Index')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "tfidf_similarity_by_rule_based.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualisierungen gespeichert unter: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantische Ähnlichkeitsanalyse für Gerichtsurteile-Datensatz")
    parser.add_argument("--input", "-i", type=str,
                        required=True, help="Pfad zur Eingabe-CSV-Datei")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="outputs", help="Ausgabeverzeichnis")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Visualisierungen erstellen")
    parser.add_argument("--max_features", type=int, default=10000,
                        help="Maximale Anzahl an TF-IDF Features")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Textvorverarbeitung überspringen")
    args = parser.parse_args()

    # Datensatz laden
    df = load_dataset(args.input)

    # Semantische Ähnlichkeit berechnen
    df_with_similarity = compute_tfidf_similarity(
        df,
        max_features=args.max_features,
        preprocess=not args.no_preprocess,
        save_visualizations=args.visualize,
        output_dir=args.output_dir
    )

    # Ergebnisse speichern
    save_results(df_with_similarity, args.output_dir,
                 "similarity_analysis.csv")

    # Bereinigte Version speichern (Optional)
    cleaned_df = df_with_similarity[~df_with_similarity['is_dissimilar_tfidf']].copy(
    )
    save_results(cleaned_df, args.output_dir, "similarity_cleaned.csv")

    print(
        f"Bereinigter Datensatz: {len(cleaned_df)} Einträge (von ursprünglich {len(df)})")
    print(
        f"Entfernt: {len(df) - len(cleaned_df)} Einträge ({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()
