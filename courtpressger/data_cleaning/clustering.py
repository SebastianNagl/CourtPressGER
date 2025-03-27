"""
Unüberwachte Clustering-Verfahren zur Bereinigung des Datensatzes.

Implementiert verschiedene Clustering-Methoden zur Identifikation von
Gruppen ähnlicher Pressemitteilungen und Erkennung von Ankündigungsclustern.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

from courtpressger.data_cleaning.utils import (
    load_dataset, save_results, load_spacy_model,
    batch_preprocess_texts, GPU_AVAILABLE
)

# Importiere GPU-beschleunigte Versionen falls verfügbar
if GPU_AVAILABLE:
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.decomposition import PCA as cuPCA
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import DBSCAN as cuDBSCAN


def perform_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    n_components: int = 50,
    max_features: int = 10000,
    min_df: int = 5,
    max_df: float = 0.85,
    use_pca: bool = True,
    algorithm: str = "kmeans",
    preprocess: bool = True,
    save_visualizations: bool = False,
    output_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Führt unüberwachtes Clustering auf den Pressemitteilungen durch.

    Args:
        df: DataFrame mit Pressemitteilungen
        n_clusters: Anzahl der Cluster (nur für K-Means)
        n_components: Anzahl der PCA-Komponenten für Dimensionsreduktion
        max_features: Maximale Anzahl an TF-IDF Features
        min_df: Minimale Dokument-Frequenz für TF-IDF
        max_df: Maximale Dokument-Frequenz für TF-IDF
        use_pca: Ob PCA zur Dimensionsreduktion verwendet werden soll
        algorithm: Clustering-Algorithmus ('kmeans' oder 'dbscan')
        preprocess: Ob Texte vorverarbeitet werden sollen
        save_visualizations: Ob Visualisierungen gespeichert werden sollen
        output_dir: Ausgabeverzeichnis

    Returns:
        DataFrame mit Cluster-Labels
    """
    print("\n## Unüberwachte Verfahren (Clustering/Topic Modeling)")

    # Vorverarbeitung der Texte
    if preprocess:
        print("Führe Text-Vorverarbeitung durch...")
        nlp = load_spacy_model()
        df['preprocessed_summary'] = batch_preprocess_texts(df['summary'], nlp)
        texts = df['preprocessed_summary']
    else:
        texts = df['summary'].fillna('')

    # TF-IDF Vektorisierung
    print("Berechne TF-IDF Vektoren für Clustering...")

    if GPU_AVAILABLE:
        tfidf_vectorizer = cuTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    else:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    # Anwenden des Vektorisierers
    vectors = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"Vokabulargröße: {len(feature_names)}")

    # Dimensionsreduktion mit PCA
    if use_pca:
        print(
            f"Reduziere Dimensionalität mit PCA auf {n_components} Komponenten...")

        if GPU_AVAILABLE:
            # GPU-beschleunigte PCA
            pca = cuPCA(n_components=n_components, copy=True)
            vectors_dense = vectors.toarray().astype(cp.float32)
            vectors_reduced = pca.fit_transform(vectors_dense)
            explained_variance = sum(pca.explained_variance_ratio_)
        else:
            # CPU-basierte PCA
            pca = PCA(n_components=n_components)
            vectors_dense = vectors.toarray()
            vectors_reduced = pca.fit_transform(vectors_dense)
            explained_variance = sum(pca.explained_variance_ratio_)

        print(f"Erklärte Varianz durch PCA: {explained_variance:.2f}")
    else:
        # Wenn keine PCA gewünscht ist, verwende die vollständigen Vektoren
        if GPU_AVAILABLE:
            vectors_reduced = vectors.toarray().astype(cp.float32)
        else:
            vectors_reduced = vectors.toarray()

    # Clustering durchführen
    if algorithm.lower() == 'kmeans':
        print(f"Führe K-Means Clustering mit {n_clusters} Clustern durch...")

        if GPU_AVAILABLE:
            # GPU-beschleunigtes K-Means
            kmeans = cuKMeans(n_clusters=n_clusters,
                              random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors_reduced)
        else:
            # CPU-basiertes K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors_reduced)

    elif algorithm.lower() == 'dbscan':
        print("Führe DBSCAN Clustering durch...")

        if GPU_AVAILABLE:
            # GPU-beschleunigtes DBSCAN
            dbscan = cuDBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(vectors_reduced)
        else:
            # CPU-basiertes DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(vectors_reduced)

        # Konvertiere -1 (Noise) in sequentielle Cluster-IDs
        if -1 in cluster_labels:
            max_label = max(cluster_labels)
            cluster_labels[cluster_labels == -1] = max_label + 1

    else:
        raise ValueError(f"Unbekannter Clustering-Algorithmus: {algorithm}")

    # Speichere die Cluster-Labels im DataFrame
    df['cluster'] = cluster_labels

    # Analysiere die Cluster
    print("Analysiere Cluster...")
    cluster_distribution = df['cluster'].value_counts().sort_index()
    print("Verteilung der Einträge pro Cluster:")
    for cluster_id, count in cluster_distribution.items():
        print(
            f"Cluster {cluster_id}: {count} Einträge ({count/len(df)*100:.2f}%)")

    # Identifiziere potenzielle Ankündigungscluster, wenn regelbasierte Labels verfügbar sind
    if 'is_announcement_rule' in df.columns:
        print("\nAnalyse der Cluster im Vergleich zu regelbasierten Erkennungen:")
        cluster_analysis = df.groupby(
            'cluster')['is_announcement_rule'].mean().sort_values(ascending=False)
        print("Anteil der Ankündigungen in jedem Cluster (regelbasiert):")
        for cluster_id, ratio in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {ratio:.2%}")

        # Bestimme den Cluster mit dem höchsten Anteil an Ankündigungen
        announcement_cluster = cluster_analysis.index[0]
        print(
            f"\nCluster {announcement_cluster} hat den höchsten Anteil an potenziellen Ankündigungen ({cluster_analysis.iloc[0]:.2%})")

        # Markiere Einträge im Ankündigungscluster
        df['is_irrelevant_cluster'] = df['cluster'] == announcement_cluster

        # Auswertung der Cluster-Methode
        cluster_irrelevant_count = df['is_irrelevant_cluster'].sum()
        print(
            f"Cluster-Methode: {cluster_irrelevant_count} von {len(df)} Einträgen ({cluster_irrelevant_count/len(df)*100:.2f}%) im Ankündigungs-Cluster")

        # Überlappung mit anderen Methoden
        if 'is_dissimilar_tfidf' in df.columns and 'is_irrelevant_ml' in df.columns:
            print("\nÜberlappung mit anderen Methoden:")
            print(
                f"- Mit regelbasierter Methode: {(df['is_announcement_rule'] & df['is_irrelevant_cluster']).sum()} Einträge")
            print(
                f"- Mit TF-IDF Ähnlichkeit: {(df['is_dissimilar_tfidf'] & df['is_irrelevant_cluster']).sum()} Einträge")
            print(
                f"- Mit ML-Klassifikator: {(df['is_irrelevant_ml'] & df['is_irrelevant_cluster']).sum()} Einträge")

    # Extrahiere die wichtigsten Wörter für jeden Cluster
    print("\nWichtigste Wörter pro Cluster:")
    cluster_keywords = extract_cluster_keywords(df, vectors, feature_names)

    # Erstelle Visualisierungen wenn gewünscht
    if save_visualizations:
        create_clustering_visualizations(
            df, vectors_reduced, output_dir, cluster_keywords)

    return df


def extract_cluster_keywords(df: pd.DataFrame, vectors, feature_names: np.ndarray, top_n: int = 10) -> Dict[int, List[str]]:
    """
    Extrahiert die charakteristischsten Wörter für jeden Cluster.

    Args:
        df: DataFrame mit Cluster-Labels
        vectors: TF-IDF Vektoren der Dokumente
        feature_names: Namen der Features (Wörter) im Vektorisierer
        top_n: Anzahl der Top-Wörter pro Cluster

    Returns:
        Dictionary mit Cluster-IDs als Schlüssel und Listen von Top-Wörtern als Werte
    """
    cluster_keywords = {}

    for cluster_id in sorted(df['cluster'].unique()):
        # Hole die Indizes der Dokumente in diesem Cluster
        cluster_docs = df[df['cluster'] == cluster_id].index

        if len(cluster_docs) > 0:
            # Berechne die durchschnittlichen TF-IDF-Werte für diesen Cluster
            if GPU_AVAILABLE:
                # Für GPU: Extrahiere Untermatrix und konvertiere zu CPU für Mittelwertberechnung
                cluster_tfidf_gpu = vectors[cluster_docs]
                cluster_tfidf = cp.asnumpy(
                    cluster_tfidf_gpu.mean(axis=0).toarray()).flatten()
            else:
                # Für CPU: Direktes Berechnen des Mittelwerts
                cluster_tfidf = vectors[cluster_docs].mean(axis=0)
                cluster_tfidf = np.asarray(cluster_tfidf).flatten()

            # Sortiere nach TF-IDF-Werten und hole die Top-Wörter
            top_indices = np.argsort(cluster_tfidf)[::-1][:top_n]
            top_words = [feature_names[idx] for idx in top_indices]

            # Speichere die Top-Wörter
            cluster_keywords[cluster_id] = top_words

            # Zeige die Top-Wörter an
            print(
                f"Cluster {cluster_id} (n={len(cluster_docs)}) - Top-Wörter: {', '.join(top_words)}")

    return cluster_keywords


def create_clustering_visualizations(
    df: pd.DataFrame,
    vectors_reduced: np.ndarray,
    output_dir: str,
    cluster_keywords: Dict[int, List[str]]
) -> None:
    """
    Erstellt Visualisierungen für die Clustering-Ergebnisse.

    Args:
        df: DataFrame mit Cluster-Labels
        vectors_reduced: Reduzierte Vektoren für Visualisierung
        output_dir: Ausgabeverzeichnis für Visualisierungen
        cluster_keywords: Dictionary mit Top-Wörtern pro Cluster
    """
    output_path = Path(output_dir)
    
    # Überprüfe, ob der Ausgabepfad "reports" enthält
    if "reports" not in str(output_path):
        # Wenn nicht, erstelle einen reports/clustering Unterordner
        parent_dir = output_path.parent if output_path.is_file() else output_path
        output_path = parent_dir / "reports" / "clustering"
    
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. PCA für 2D-Visualisierung (falls Dimensionalität > 2)
    if vectors_reduced.shape[1] > 2:
        print("Reduziere auf 2 Dimensionen für Visualisierung...")
        if GPU_AVAILABLE:
            pca_vis = cuPCA(n_components=2)
            vectors_2d = pca_vis.fit_transform(vectors_reduced)
            # Konvertiere zu CPU für Plotting
            vectors_2d = cp.asnumpy(vectors_2d)
        else:
            pca_vis = PCA(n_components=2)
            vectors_2d = pca_vis.fit_transform(vectors_reduced)
    else:
        # Wenn bereits 2D oder weniger, verwende direkt
        vectors_2d = vectors_reduced

    # 2. Cluster-Visualisierung
    plt.figure(figsize=(12, 10))

    # Erstelle Farbpalette mit ausreichend Farben
    n_clusters = len(df['cluster'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    # Plotte jeden Cluster einzeln für bessere Kontrolle
    for i, cluster_id in enumerate(sorted(df['cluster'].unique())):
        mask = df['cluster'] == cluster_id
        plt.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=[colors[i]],
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=30
        )

    plt.title('Clustering der Pressemitteilungen (PCA-reduziert)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)

    # Füge Legende hinzu, wenn nicht zu viele Cluster
    if n_clusters <= 20:
        plt.legend(loc='upper right')

    plt.savefig(output_path / "clustering_visualization.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Cluster-Größen-Visualisierung
    plt.figure(figsize=(12, 6))
    cluster_sizes = df['cluster'].value_counts().sort_index()

    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
    plt.title('Größe der Cluster')
    plt.xlabel('Cluster-ID')
    plt.ylabel('Anzahl der Einträge')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path / "cluster_sizes.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Visualisierung der Cluster-Keywords als Wordcloud
    try:
        from wordcloud import WordCloud

        # Erstelle eine Wordcloud für jeden Cluster
        for cluster_id, keywords in cluster_keywords.items():
            # Erstelle ein Dictionary mit Wörtern und Gewichten (einfach absteigend)
            word_weights = {word: len(keywords) -
                            i for i, word in enumerate(keywords)}

            # Erstelle die Wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_weights)

            # Zeige die Wordcloud an
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Häufigste Wörter in Cluster {cluster_id}')
            plt.tight_layout()

            plt.savefig(
                output_path / f"wordcloud_cluster_{cluster_id}.png", dpi=300, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("wordcloud Paket nicht installiert, überspringe Wordcloud-Visualisierung.")

    # 5. Vergleich mit anderen Methoden (falls vorhanden)
    if all(col in df.columns for col in ['is_announcement_rule', 'is_dissimilar_tfidf', 'is_irrelevant_ml']):
        plt.figure(figsize=(12, 6))

        # Erstelle DataFrame für einfachere Visualisierung
        comparison_df = pd.DataFrame({
            'Methode': [
                'Regelbasiert',
                'TF-IDF Ähnlichkeit',
                'ML-Klassifikator',
                'Clustering'
            ],
            'Anzahl': [
                df['is_announcement_rule'].sum(),
                df['is_dissimilar_tfidf'].sum(),
                df['is_irrelevant_ml'].sum(),
                df['is_irrelevant_cluster'].sum()
            ]
        })

        comparison_df['Prozent'] = comparison_df['Anzahl'] / len(df) * 100

        sns.barplot(data=comparison_df, x='Methode', y='Prozent')
        plt.title('Vergleich der verschiedenen Erkennungsmethoden')
        plt.xlabel('Methode')
        plt.ylabel('Prozent der Einträge')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path / "method_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualisierungen gespeichert unter: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clustering für Gerichtsurteile-Datensatz")
    parser.add_argument("--input", "-i", type=str,
                        required=True, help="Pfad zur Eingabe-CSV-Datei")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="outputs", help="Ausgabeverzeichnis")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Visualisierungen erstellen")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Anzahl der Cluster (für K-Means)")
    parser.add_argument("--n_components", type=int,
                        default=50, help="Anzahl der PCA-Komponenten")
    parser.add_argument("--algorithm", type=str, default="kmeans",
                        choices=["kmeans", "dbscan"], help="Clustering-Algorithmus")
    parser.add_argument("--no_pca", action="store_true",
                        help="PCA nicht verwenden")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Textvorverarbeitung überspringen")
    args = parser.parse_args()

    # Datensatz laden
    df = load_dataset(args.input)

    # Clustering durchführen
    df_with_clusters = perform_clustering(
        df,
        n_clusters=args.n_clusters,
        n_components=args.n_components,
        algorithm=args.algorithm,
        use_pca=not args.no_pca,
        preprocess=not args.no_preprocess,
        save_visualizations=args.visualize,
        output_dir=args.output_dir
    )

    # Ergebnisse speichern
    save_results(df_with_clusters, args.output_dir, "clustering_results.csv")

    # Bereinigte Version speichern (falls Ankündigungscluster identifiziert wurde)
    if 'is_irrelevant_cluster' in df_with_clusters.columns:
        cleaned_df = df_with_clusters[~df_with_clusters['is_irrelevant_cluster']].copy(
        )
        save_results(cleaned_df, args.output_dir, "clustering_cleaned.csv")

        print(
            f"Bereinigter Datensatz: {len(cleaned_df)} Einträge (von ursprünglich {len(df)})")
        print(
            f"Entfernt: {len(df) - len(cleaned_df)} Einträge ({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()
