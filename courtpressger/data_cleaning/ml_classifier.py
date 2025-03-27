"""
Überwachter Maschinelles Lernen-Klassifikator für die Datensatzbereinigung.

Trainiert einen Klassifikator zur Erkennung von irrelevanten Pressemitteilungen
basierend auf Textmerkmalen und optionalen Ergebnissen anderer Filter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union, Any

from courtpressger.data_cleaning.utils import (
    load_dataset, save_results, GPU_AVAILABLE
)

# GPU-beschleunigte Implementierung wenn verfügbar
if GPU_AVAILABLE:
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.linear_model import LogisticRegression as cuLogisticRegression


def train_classifier(
    df: pd.DataFrame,
    max_features: int = 5000,
    min_df: int = 3,
    max_df: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    use_rule_based: bool = True,
    use_similarity: bool = True,
    save_visualizations: bool = False,
    output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, Any]:
    """
    Trainiert einen überwachten ML-Klassifikator zur Erkennung irrelevanter Pressemitteilungen.

    Args:
        df: DataFrame mit Urteilen und Pressemitteilungen
        max_features: Maximale Anzahl an TF-IDF Features
        min_df: Minimale Dokument-Frequenz für TF-IDF
        max_df: Maximale Dokument-Frequenz für TF-IDF
        ngram_range: N-Gramm-Bereich für TF-IDF
        test_size: Anteil der Testdaten
        use_rule_based: Ob regelbasierte Filter als Label verwendet werden sollen
        use_similarity: Ob Ähnlichkeitsfilter als Label verwendet werden sollen
        save_visualizations: Ob Visualisierungen gespeichert werden sollen
        output_dir: Ausgabeverzeichnis

    Returns:
        Tuple mit DataFrame (mit ML-Klassifikationsergebnissen) und trainiertem Klassifikator
    """
    print("\n## Überwachtes Machine Learning (Klassifikation)")

    # Erstelle Labels für das Training basierend auf vorherigen Methoden
    print("Erstelle Labels für das Training basierend auf vorherigen Methoden...")

    # Überprüfe, ob die benötigten Spalten vorhanden sind
    required_columns = []
    if use_rule_based:
        required_columns.append('is_announcement_rule')
    if use_similarity:
        required_columns.append('is_dissimilar_tfidf')
        required_columns.append('tfidf_similarity')

    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(
            f"Warnung: Folgende benötigte Spalten fehlen im DataFrame: {missing_columns}")
        print("Erstelle Ersatz-Spalten mit Standardwerten.")

        for col in missing_columns:
            if col == 'is_announcement_rule':
                print(
                    "Regelbasierte Klassifikation nicht verfügbar, verwende Dummy-Werte.")
                df['is_announcement_rule'] = False
            elif col == 'is_dissimilar_tfidf':
                print(
                    "TF-IDF Ähnlichkeitsklassifikation nicht verfügbar, verwende Dummy-Werte.")
                df['is_dissimilar_tfidf'] = False
            elif col == 'tfidf_similarity':
                print("TF-IDF Ähnlichkeitswerte nicht verfügbar, verwende Dummy-Werte.")
                df['tfidf_similarity'] = 1.0  # Alle Paare vollständig ähnlich

    # Definiere Schwellenwert für Ähnlichkeit, falls diese Spalte verwendet wird
    similarity_threshold = df['tfidf_similarity'].quantile(
        0.1) if 'tfidf_similarity' in df.columns else 0.0

    # Wir betrachten Einträge als "irrelevant" (Label 1), wenn sie entweder
    # durch die regelbasierte Methode als Ankündigungen erkannt wurden
    # oder eine sehr niedrige semantische Ähnlichkeit aufweisen
    if use_rule_based and use_similarity:
        df['is_irrelevant'] = ((df['is_announcement_rule']) |
                               (df['tfidf_similarity'] < similarity_threshold * 0.8))
    elif use_rule_based:
        df['is_irrelevant'] = df['is_announcement_rule']
    elif use_similarity:
        df['is_irrelevant'] = df['tfidf_similarity'] < similarity_threshold * 0.8
    else:
        # Keine Filter verfügbar, verwende einen Dummywert (10% der Daten als irrelevant)
        print("Keine Filter für Labels verfügbar, verwende zufällige Auswahl als Dummy-Labels (10%).")
        df['is_irrelevant'] = np.random.choice(
            [True, False],
            size=len(df),
            p=[0.1, 0.9]  # 10% als irrelevant markieren
        )

    # Balancieren der Daten (optional)
    irrelevant_count = df['is_irrelevant'].sum()
    print(
        f"Anzahl der als irrelevant markierten Einträge: {irrelevant_count} ({irrelevant_count/len(df)*100:.2f}%)")

    # Features für das Training erstellen (TF-IDF auf den Pressemitteilungen)
    print("Erstelle Features für das Klassifikationsmodell...")

    if GPU_AVAILABLE:
        classifier_vectorizer = cuTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
    else:
        classifier_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )

    # Wir nutzen die Originaltexte (nicht die vorverarbeiteten)
    # um auch strukturelle Merkmale zu erfassen
    X = classifier_vectorizer.fit_transform(df['summary'].fillna(''))
    y = df['is_irrelevant'].astype(int)

    # Aufteilen in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(
        f"Trainingsdaten: {X_train.shape[0]} Einträge, Testdaten: {X_test.shape[0]} Einträge")
    print(f"Klassen-Verteilung im Training: {Counter(y_train)}")

    # Logistische Regression für Klassifikation
    print("Trainiere Logistic Regression Klassifikator...")

    if GPU_AVAILABLE:
        # GPU-beschleunigte Logistische Regression
        classifier = cuLogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    else:
        # CPU-basierte Logistische Regression
        classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

    classifier.fit(X_train, y_train)

    # Modell evaluieren
    y_pred = classifier.predict(X_test)
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]

    # Zeige Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Klassifikationsreport zeigen
    print("\nKlassifikationsreport:")
    print(classification_report(y_test, y_pred))

    # Die wichtigsten Merkmale für jede Klasse anzeigen
    print("\nWichtigste Merkmale für die Erkennung von irrelevanten Pressemitteilungen:")
    if hasattr(classifier, 'coef_'):
        feature_names = classifier_vectorizer.get_feature_names_out()
        # Sortiere nach absoluten Koeffizienten (größte zuerst)
        sorted_coef_indices = np.argsort(np.abs(classifier.coef_[0]))[::-1]

        # Top N positive und negative Features
        n_features = 20

        # Features, die für "irrelevant" sprechen (positive Koeffizienten)
        print("Features, die für 'irrelevant' sprechen:")
        pos_indices = [idx for idx in sorted_coef_indices if classifier.coef_[
            0][idx] > 0][:n_features]
        for idx in pos_indices:
            print(f"  {feature_names[idx]}: {classifier.coef_[0][idx]:.4f}")

        # Features, die für "relevant" sprechen (negative Koeffizienten)
        print("\nFeatures, die für 'relevant' sprechen:")
        neg_indices = [idx for idx in sorted_coef_indices if classifier.coef_[
            0][idx] < 0][:n_features]
        for idx in neg_indices:
            print(f"  {feature_names[idx]}: {classifier.coef_[0][idx]:.4f}")

    # Anwenden des Modells auf den gesamten Datensatz
    print("\nWende trainiertes Modell auf den gesamten Datensatz an...")
    df['irrelevant_prob'] = classifier.predict_proba(X)[:, 1]
    df['is_irrelevant_ml'] = classifier.predict(X)

    # Auswertung der ML-Methode
    ml_irrelevant_count = df['is_irrelevant_ml'].sum()
    print(
        f"ML-Klassifikator: {ml_irrelevant_count} von {len(df)} Einträgen ({ml_irrelevant_count/len(df)*100:.2f}%) wurden als irrelevant klassifiziert.")

    # Vergleich mit vorherigen Methoden
    if use_rule_based:
        print(
            f"- Mit regelbasierter Methode: {(df['is_announcement_rule'] & df['is_irrelevant_ml']).sum()} Einträge")
    if use_similarity:
        print(
            f"- Mit TF-IDF Ähnlichkeit: {(df['is_dissimilar_tfidf'] & df['is_irrelevant_ml']).sum()} Einträge")

    # Erstelle Visualisierungen wenn gewünscht
    if save_visualizations:
        create_ml_visualizations(
            df, cm, y_test, y_pred, y_pred_prob, output_dir)

    # Rückgabe des DataFrames mit ML-Ergebnissen und des Klassifikators für spätere Verwendung
    return df, (classifier, classifier_vectorizer)


def create_ml_visualizations(
    df: pd.DataFrame,
    confusion_mat: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    output_dir: str
) -> None:
    """
    Erstellt Visualisierungen für die ML-Klassifikation.

    Args:
        df: DataFrame mit Klassifikationsergebnissen
        confusion_mat: Confusion Matrix
        y_test: Wahre Labels der Testdaten
        y_pred: Vorhergesagte Labels der Testdaten
        y_pred_prob: Vorhergesagte Wahrscheinlichkeiten der Testdaten
        output_dir: Ausgabeverzeichnis für Visualisierungen
    """
    output_path = Path(output_dir)
    
    # Überprüfe, ob der Ausgabepfad "reports" enthält
    if "reports" not in str(output_path):
        # Wenn nicht, erstelle einen reports/ml_classifier Unterordner
        parent_dir = output_path.parent if output_path.is_file() else output_path
        output_path = parent_dir / "reports" / "ml_classifier"
    
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. Confusion Matrix visualisieren
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relevant', 'Irrelevant'],
                yticklabels=['Relevant', 'Irrelevant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Vorhergesagt')
    plt.ylabel('Tatsächlich')
    plt.tight_layout()
    plt.savefig(output_path / "ml_confusion_matrix.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC-Kurve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC-Kurve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsch-Positiv-Rate')
    plt.ylabel('Richtig-Positiv-Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "ml_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Verteilung der Wahrscheinlichkeiten
    plt.figure(figsize=(10, 6))
    sns.histplot(df['irrelevant_prob'], bins=50, kde=True)
    plt.title('Verteilung der ML-Wahrscheinlichkeiten für "irrelevant"')
    plt.xlabel('Wahrscheinlichkeit')
    plt.ylabel('Anzahl der Einträge')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "ml_probability_distribution.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Vergleich mit anderen Methoden (falls vorhanden)
    if 'is_announcement_rule' in df.columns and 'is_dissimilar_tfidf' in df.columns:
        method_counts = pd.DataFrame({
            'Methode': [
                'Regelbasiert',
                'TF-IDF Ähnlichkeit',
                'ML-Klassifikator',
                'Überlappung: Regel + ML',
                'Überlappung: TF-IDF + ML',
                'Überlappung: Alle'
            ],
            'Anzahl': [
                df['is_announcement_rule'].sum(),
                df['is_dissimilar_tfidf'].sum(),
                df['is_irrelevant_ml'].sum(),
                (df['is_announcement_rule'] & df['is_irrelevant_ml']).sum(),
                (df['is_dissimilar_tfidf'] & df['is_irrelevant_ml']).sum(),
                (df['is_announcement_rule'] & df['is_dissimilar_tfidf']
                 & df['is_irrelevant_ml']).sum()
            ]
        })

        method_counts['Prozent'] = method_counts['Anzahl'] / len(df) * 100

        plt.figure(figsize=(12, 6))
        sns.barplot(data=method_counts, x='Methode', y='Prozent')
        plt.title('Vergleich der verschiedenen Erkennungsmethoden')
        plt.xlabel('Methode')
        plt.ylabel('Prozent der Einträge')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "ml_method_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualisierungen gespeichert unter: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ML-Klassifikator für Gerichtsurteile-Datensatz")
    parser.add_argument("--input", "-i", type=str,
                        required=True, help="Pfad zur Eingabe-CSV-Datei")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="outputs", help="Ausgabeverzeichnis")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Visualisierungen erstellen")
    parser.add_argument("--max_features", type=int, default=5000,
                        help="Maximale Anzahl an TF-IDF Features")
    parser.add_argument("--test_size", type=float,
                        default=0.2, help="Anteil der Testdaten")
    parser.add_argument("--no_rule_based", action="store_true",
                        help="Regelbasierte Filter nicht verwenden")
    parser.add_argument("--no_similarity", action="store_true",
                        help="Ähnlichkeitsfilter nicht verwenden")
    parser.add_argument("--model_output", type=str,
                        help="Pfad zum Speichern des trainierten Modells")
    args = parser.parse_args()

    # Datensatz laden
    df = load_dataset(args.input)

    # ML-Klassifikator trainieren
    df_with_ml, model_artifacts = train_classifier(
        df,
        max_features=args.max_features,
        test_size=args.test_size,
        use_rule_based=not args.no_rule_based,
        use_similarity=not args.no_similarity,
        save_visualizations=args.visualize,
        output_dir=args.output_dir
    )

    # Ergebnisse speichern
    save_results(df_with_ml, args.output_dir, "ml_classification.csv")

    # Bereinigte Version speichern (Optional)
    cleaned_df = df_with_ml[~df_with_ml['is_irrelevant_ml']].copy()
    save_results(cleaned_df, args.output_dir, "ml_cleaned.csv")

    print(
        f"Bereinigter Datensatz: {len(cleaned_df)} Einträge (von ursprünglich {len(df)})")
    print(
        f"Entfernt: {len(df) - len(cleaned_df)} Einträge ({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")

    # Modell speichern, wenn gewünscht
    if args.model_output:
        try:
            import joblib
            model_dir = Path(args.model_output).parent
            model_dir.mkdir(exist_ok=True, parents=True)
            joblib.dump(model_artifacts, args.model_output)
            print(f"Modell gespeichert unter: {args.model_output}")
        except Exception as e:
            print(f"Fehler beim Speichern des Modells: {e}")


if __name__ == "__main__":
    main()
