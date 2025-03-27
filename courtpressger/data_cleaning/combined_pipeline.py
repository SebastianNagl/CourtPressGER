"""
Kombinierte Pipeline für die Datensatzbereinigung.

Führt alle Bereinigungsmethoden sequentiell aus und kombiniert deren Ergebnisse
zu einem optimalen bereinigten Datensatz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from courtpressger.data_cleaning.utils import (
    load_dataset, save_results, GPU_AVAILABLE
)
from courtpressger.data_cleaning.rule_based import apply_rule_based_filters
from courtpressger.data_cleaning.semantic_similarity import compute_tfidf_similarity
from courtpressger.data_cleaning.ml_classifier import train_classifier
from courtpressger.data_cleaning.clustering import perform_clustering


def run_combined_pipeline(
    df: pd.DataFrame,
    min_votes: int = 2,
    save_visualizations: bool = False,
    output_dir: str = "outputs",
    skip_rule_based: bool = False,
    skip_similarity: bool = False,
    skip_ml: bool = False,
    skip_clustering: bool = False
) -> pd.DataFrame:
    """
    Führt alle Bereinigungsmethoden aus und kombiniert ihre Ergebnisse.

    Args:
        df: DataFrame mit Gerichtsurteilen und Pressemitteilungen
        min_votes: Minimale Anzahl an Methoden, die einen Eintrag als irrelevant einstufen müssen
        save_visualizations: Ob Visualisierungen gespeichert werden sollen
        output_dir: Ausgabeverzeichnis
        skip_rule_based: Ob regelbasierte Filterung übersprungen werden soll
        skip_similarity: Ob Ähnlichkeitsberechnung übersprungen werden soll
        skip_ml: Ob ML-Klassifikation übersprungen werden soll
        skip_clustering: Ob Clustering übersprungen werden soll

    Returns:
        DataFrame mit kombinierten Bereinigungsergebnissen
    """
    print("## Kombinierte Bereinigungspipeline")

    # Erstelle Ausgabeordner
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Erstelle reports Ordner für Visualisierungen
    reports_dir = output_path / "reports"
    reports_dir.mkdir(exist_ok=True, parents=True)

    # 1. Regelbasierte Filterung
    if not skip_rule_based:
        print("\n### 1. Regelbasierte Filterung")
        df = apply_rule_based_filters(
            df,
            save_visualizations=save_visualizations,
            output_dir=str(reports_dir / "rule_based")
        )
    else:
        print("\n### 1. Regelbasierte Filterung (übersprungen)")
        if 'is_announcement_rule' not in df.columns:
            df['is_announcement_rule'] = False

    # 2. Semantische Ähnlichkeit
    if not skip_similarity:
        print("\n### 2. Semantische Ähnlichkeit")
        df = compute_tfidf_similarity(
            df,
            save_visualizations=save_visualizations,
            output_dir=str(reports_dir / "similarity")
        )
    else:
        print("\n### 2. Semantische Ähnlichkeit (übersprungen)")
        if 'is_dissimilar_tfidf' not in df.columns:
            df['is_dissimilar_tfidf'] = False
            df['tfidf_similarity'] = 1.0  # Alle vollständig ähnlich

    # 3. ML-Klassifikation
    if not skip_ml:
        print("\n### 3. ML-Klassifikation")
        df, _ = train_classifier(
            df,
            save_visualizations=save_visualizations,
            output_dir=str(reports_dir / "ml_classifier")
        )
    else:
        print("\n### 3. ML-Klassifikation (übersprungen)")
        if 'is_irrelevant_ml' not in df.columns:
            df['is_irrelevant_ml'] = False
            df['irrelevant_prob'] = 0.0  # Keine irrelevant

    # 4. Clustering
    if not skip_clustering:
        print("\n### 4. Clustering")
        df = perform_clustering(
            df,
            save_visualizations=save_visualizations,
            output_dir=str(reports_dir / "clustering")
        )
    else:
        print("\n### 4. Clustering (übersprungen)")
        if 'is_irrelevant_cluster' not in df.columns:
            df['is_irrelevant_cluster'] = False
            df['cluster'] = 0  # Alle im selben Cluster

    # 5. Kombinierter Ansatz
    print("\n### 5. Kombinierte Bereinigung")

    # Zähle, von wie vielen Methoden ein Eintrag als irrelevant eingestuft wurde
    df['irrelevant_votes'] = (
        df['is_announcement_rule'].astype(int) +
        df['is_dissimilar_tfidf'].astype(int) +
        df['is_irrelevant_ml'].astype(int) +
        df['is_irrelevant_cluster'].astype(int)
    )

    # Analysiere die Verteilung der Stimmen
    vote_counts = df['irrelevant_votes'].value_counts().sort_index()
    print("Verteilung der 'irrelevant'-Stimmen:")
    for votes, count in vote_counts.items():
        print(f"{votes} Methode(n): {count} Einträge ({count/len(df)*100:.2f}%)")

    # Wir betrachten Einträge als irrelevant, wenn mindestens N Methoden sie als irrelevant eingestuft haben
    df['is_irrelevant_combined'] = df['irrelevant_votes'] >= min_votes

    # Auswertung des kombinierten Ansatzes
    combined_irrelevant_count = df['is_irrelevant_combined'].sum()
    print(f"\nKombinierter Ansatz (≥{min_votes} Methoden): {combined_irrelevant_count} von {len(df)} Einträgen ({combined_irrelevant_count/len(df)*100:.2f}%) als irrelevant identifiziert.")

    # Erstelle Visualisierungen für die kombinierte Methode
    if save_visualizations:
        create_combined_visualizations(df, min_votes, output_dir)

    return df


def create_combined_visualizations(df: pd.DataFrame, min_votes: int, output_dir: Path) -> None:
    """
    Erstellt Visualisierungen für die kombinierte Bereinigungsmethode.

    Args:
        df: DataFrame mit kombinierten Bereinigungsergebnissen
        min_votes: Verwendeter Schwellenwert für Mindestanzahl an Stimmen
        output_dir: Ausgabeverzeichnis für Visualisierungen
    """
    # Überprüfe, ob der Ausgabepfad "reports" enthält
    if "reports" not in str(output_dir):
        # Wenn nicht, erstelle einen reports/combined Unterordner
        parent_dir = output_dir.parent if output_dir.is_file() else output_dir
        output_dir = parent_dir / "reports"
    
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(exist_ok=True, parents=True)

    # 1. Venn-Diagramm der Methoden
    try:
        plt.figure(figsize=(10, 10))
        from matplotlib_venn import venn4, venn4_circles

        # Erstelle Sets der irrelevanten Einträge für jede Methode
        rule_based_set = set(df[df['is_announcement_rule']].index)
        similarity_set = set(df[df['is_dissimilar_tfidf']].index)
        ml_set = set(df[df['is_irrelevant_ml']].index)
        cluster_set = set(df[df['is_irrelevant_cluster']].index)

        # Zeichne das Venn-Diagramm
        venn = venn4(
            [rule_based_set, similarity_set, ml_set, cluster_set],
            ('Regelbasiert', 'Semantische Ähnlichkeit',
             'ML-Klassifikation', 'Clustering')
        )
        venn4_circles(
            [rule_based_set, similarity_set, ml_set, cluster_set],
            linestyle='solid'
        )
        plt.title('Überlappung der erkannten irrelevanten Einträge')
        plt.savefig(combined_dir / "venn_diagram.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    except ImportError:
        print("matplotlib-venn nicht installiert, überspringe Venn-Diagramm.")

    # 2. Verteilung der Stimmen
    plt.figure(figsize=(10, 6))
    vote_counts = df['irrelevant_votes'].value_counts().sort_index()
    sns.barplot(x=vote_counts.index, y=vote_counts.values)
    plt.axvline(x=min_votes-0.5, color='red', linestyle='--',
                label=f'Schwellenwert (≥{min_votes} Stimmen)')
    plt.legend()
    plt.title('Verteilung der "irrelevant"-Stimmen pro Eintrag')
    plt.xlabel('Anzahl der Methoden, die den Eintrag als irrelevant einstufen')
    plt.ylabel('Anzahl der Einträge')
    plt.grid(True, alpha=0.3)
    plt.savefig(combined_dir / "vote_distribution.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Vergleich der einzelnen vs. kombinierten Methode
    plt.figure(figsize=(12, 6))

    # Erstelle DataFrame für einfachere Visualisierung
    comparison_df = pd.DataFrame({
        'Methode': [
            'Regelbasiert',
            'Semantische Ähnlichkeit',
            'ML-Klassifikation',
            'Clustering',
            f'Kombiniert (≥{min_votes} Methoden)'
        ],
        'Anzahl': [
            df['is_announcement_rule'].sum(),
            df['is_dissimilar_tfidf'].sum(),
            df['is_irrelevant_ml'].sum(),
            df['is_irrelevant_cluster'].sum(),
            df['is_irrelevant_combined'].sum()
        ]
    })

    comparison_df['Prozent'] = comparison_df['Anzahl'] / len(df) * 100

    sns.barplot(data=comparison_df, x='Methode', y='Prozent')
    plt.title('Vergleich der verschiedenen Erkennungsmethoden')
    plt.xlabel('Methode')
    plt.ylabel('Prozent der Einträge')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(combined_dir / "method_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Kombinierte Visualisierungen gespeichert unter: {combined_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Kombinierte Bereinigungspipeline für Gerichtsurteile-Datensatz")
    parser.add_argument("--input", "-i", type=str,
                        required=True, help="Pfad zur Eingabe-CSV-Datei")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="outputs", help="Ausgabeverzeichnis")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Visualisierungen erstellen")
    parser.add_argument("--min_votes", type=int, default=2,
                        help="Minimale Anzahl an Methoden für Kombination")
    parser.add_argument("--skip_rule_based", action="store_true",
                        help="Regelbasierte Filterung überspringen")
    parser.add_argument("--skip_similarity", action="store_true",
                        help="Ähnlichkeitsberechnung überspringen")
    parser.add_argument("--skip_ml", action="store_true",
                        help="ML-Klassifikation überspringen")
    parser.add_argument("--skip_clustering",
                        action="store_true", help="Clustering überspringen")
    args = parser.parse_args()

    # Datensatz laden
    df = load_dataset(args.input)

    # Kombinierte Pipeline ausführen
    df_combined = run_combined_pipeline(
        df,
        min_votes=args.min_votes,
        save_visualizations=args.visualize,
        output_dir=args.output_dir,
        skip_rule_based=args.skip_rule_based,
        skip_similarity=args.skip_similarity,
        skip_ml=args.skip_ml,
        skip_clustering=args.skip_clustering
    )

    # Ergebnisse speichern
    save_results(df_combined, args.output_dir, "combined_results.csv")

    # Speichere die Analyseergebnisse mit erweiterten Metriken für alle Filtermethoden
    detailed_columns = [
        'id', 'summary',
        # Ergebnisse der verschiedenen Methoden
        'is_announcement_rule', 'is_dissimilar_tfidf',
        'is_irrelevant_ml', 'is_irrelevant_cluster',
        # Zusätzliche Metriken
        'tfidf_similarity', 'irrelevant_prob',
        # Kombinierte Metriken
        'irrelevant_votes', 'is_irrelevant_combined'
    ]

    # Wähle nur Spalten aus, die tatsächlich im DataFrame existieren
    available_columns = [
        col for col in detailed_columns if col in df_combined.columns]
    detailed_results = df_combined[available_columns]
    save_results(detailed_results, args.output_dir, "detailed_analysis.csv")

    # Bereinigte Version speichern
    cleaned_df = df_combined[~df_combined['is_irrelevant_combined']].copy()
    save_results(cleaned_df, args.output_dir, "final_cleaned.csv")

    print(
        f"\nBereinigter Datensatz: {len(cleaned_df)} Einträge (von ursprünglich {len(df)})")
    print(
        f"Entfernt: {len(df) - len(cleaned_df)} Einträge ({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()
