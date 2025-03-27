"""
Regelbasierte Filter für die Bereinigung des Datensatzes.

Implementiert verschiedene regelbasierte Ansätze zur Erkennung von Ankündigungen 
und nicht urteilsbezogenen Mitteilungen.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, Any, List

from courtpressger.data_cleaning.utils import load_dataset, save_results


def is_announcement_rule_based(row: pd.Series) -> bool:
    """
    Erkennt Ankündigungen und nicht urteilsbezogene Mitteilungen anhand von Keywords und Mustern.

    Args:
        row: Zeile des DataFrames mit 'summary' Spalte

    Returns:
        True, wenn es sich vermutlich um eine Ankündigung handelt, sonst False
    """
    if pd.isna(row['summary']):
        return False

    summary = str(row['summary']).lower()

    # 1. Keywords, die auf Ankündigungen hindeuten
    future_indicators = [
        'ankündigung', 'terminankündigung', 'terminhinweis', 'hinweis auf termin',
        'wird verhandelt', 'wird verhandeln', 'wird.*stattfinden', 'findet statt',
        'einladung', 'pressetermin', 'pressekonferenz', 'veranstaltung',
        'wochenvorschau', 'jahrespressekonferenz', 'pressegespräch', 'rundgang',
        'beginnt am', 'laden ein', 'lädt ein', 'sitzung vom'
    ]

    # 2. Regex-Muster für Datumsangaben in der Zukunft (relativ zum Erstelldatum)
    date_patterns = [
        r'am\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}',  # "am 15.05.2023"
        r'vom\s+\d{1,2}\.\s*\d{1,2}\.',         # "vom 15.05."
        r'terminiert auf den \d{1,2}',           # "terminiert auf den 15"
        r'in der kommenden woche',               # zukünftige Verweise
        r'in der nächsten woche',
        r'demnächst'
    ]

    # 3. Überschriften/Anfänge, die auf Ankündigungen hindeuten
    headline_indicators = [
        'presseinformation', 'information für die presse',
        'medieninformation', 'zur information',
        'terminvorschau', 'jahresbericht', 'geschäftsbericht', 'tätigkeitsbericht',
        'stellenausschreibung', 'personelle veränderungen', 'neuer präsident'
    ]

    # 4. Prüfen auf Keywords am Anfang des Textes (größeres Gewicht)
    summary_start = summary[:100]
    headline_match = any(
        indicator in summary_start for indicator in headline_indicators)

    # 5. Prüfen auf allgemeine Ankündigungskeywords im gesamten Text
    keyword_match = any(re.search(r'\b' + re.escape(indicator) + r'\b', summary, re.IGNORECASE)
                        for indicator in future_indicators)

    # 6. Prüfen auf Datumsmuster
    date_match = any(re.search(pattern, summary, re.IGNORECASE)
                     for pattern in date_patterns)

    # Kombinierte Entscheidung mit unterschiedlicher Gewichtung
    if headline_match:
        return True  # Überschriften/Anfänge sind starke Indikatoren
    elif keyword_match and date_match:
        return True  # Kombination aus Keyword und Datumsmuster
    elif keyword_match and ('mündliche verhandlung' in summary or 'termin' in summary):
        return True  # Spezifische Kombinationen

    return False  # Default: keine Ankündigung


def analyze_rule_based_filters(text: str) -> Dict[str, Any]:
    """
    Analysiert einen Text mit verschiedenen regelbasierten Filtern und gibt deren einzelne Ergebnisse zurück.

    Args:
        text: Zu analysierender Text

    Returns:
        Dictionary mit Ergebnissen der einzelnen Filterkriterien
    """
    if pd.isna(text):
        return {
            'future_indicators': False,
            'date_patterns': False,
            'headline_indicators': False,
            'headline_date': False,
            'future_indicators_score': 0.0,
            'date_patterns_score': 0.0,
            'headline_indicators_score': 0.0,
            'combined_rule_score': 0.0
        }

    text = str(text).lower()
    results = {}

    # 1. Keywords, die auf Ankündigungen hindeuten
    future_indicators = [
        'ankündigung', 'terminankündigung', 'terminhinweis', 'hinweis auf termin',
        'wird verhandelt', 'wird verhandeln', 'wird.*stattfinden', 'findet statt',
        'einladung', 'pressetermin', 'pressekonferenz', 'veranstaltung',
        'wochenvorschau', 'jahrespressekonferenz', 'pressegespräch', 'rundgang',
        'beginnt am', 'laden ein', 'lädt ein', 'sitzung vom'
    ]

    # Zähle, wie viele der Future Indicators gefunden wurden
    future_matches = sum(1 for indicator in future_indicators if re.search(
        r'\b' + re.escape(indicator) + r'\b', text, re.IGNORECASE))
    future_indicators_score = min(
        1.0, future_matches / len(future_indicators) * 3)  # Skaliert, max 1.0
    results['future_indicators'] = future_matches > 0
    results['future_indicators_score'] = future_indicators_score

    # 2. Datumsmuster
    date_patterns = [
        r'am\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}',  # "am 15.05.2023"
        r'vom\s+\d{1,2}\.\s*\d{1,2}\.',         # "vom 15.05."
        r'terminiert auf den \d{1,2}',           # "terminiert auf den 15"
        r'in der kommenden woche',               # zukünftige Verweise
        r'in der nächsten woche',
        r'demnächst'
    ]

    # Zähle, wie viele der Date Patterns gefunden wurden
    date_matches = sum(1 for pattern in date_patterns if re.search(
        pattern, text, re.IGNORECASE))
    date_patterns_score = min(
        1.0, date_matches / len(date_patterns) * 2)  # Skaliert, max 1.0
    results['date_patterns'] = date_matches > 0
    results['date_patterns_score'] = date_patterns_score

    # 3. Überschriften/Anfänge, die auf Ankündigungen hindeuten
    headline_indicators = [
        'pressemitteilung nr', 'presseinformation', 'information für die presse',
        'medieninformation', 'zur information', 'mündliche verhandlung',
        'terminvorschau', 'jahresbericht', 'geschäftsbericht', 'tätigkeitsbericht',
        'stellenausschreibung', 'personelle veränderungen', 'neuer präsident'
    ]

    # Prüfe speziell den Textanfang (erste 100 Zeichen)
    text_start = text[:100]
    headline_matches = sum(
        1 for indicator in headline_indicators if indicator in text_start)
    headline_indicators_score = min(
        1.0, headline_matches / len(headline_indicators) * 3)  # Skaliert, max 1.0
    results['headline_indicators'] = headline_matches > 0
    results['headline_indicators_score'] = headline_indicators_score

    # 4. Kombination aus Überschrift und Datum
    results['headline_date'] = results['headline_indicators'] and results['date_patterns']

    # 5. Berechne einen kombinierten Score für die regelbasierte Methode
    # Gewichtung: Überschrift (0.5), Future Indicators (0.3), Date Patterns (0.2)
    results['combined_rule_score'] = (
        0.5 * headline_indicators_score +
        0.3 * future_indicators_score +
        0.2 * date_patterns_score
    )

    return results


def apply_rule_based_filters(df: pd.DataFrame, save_visualizations: bool = False, output_dir: str = "outputs") -> pd.DataFrame:
    """
    Wendet regelbasierte Filter auf den Datensatz an und erstellt optionale Visualisierungen.

    Args:
        df: DataFrame mit Gerichtsurteilen und Pressemitteilungen
        save_visualizations: Ob Visualisierungen gespeichert werden sollen
        output_dir: Verzeichnis für Visualisierungen

    Returns:
        DataFrame mit zusätzlichen Filter-Spalten
    """
    print("Wende regelbasierte Filter an...")

    # Einfacher Filter für Ankündigungen
    df['is_announcement_rule'] = df.apply(is_announcement_rule_based, axis=1)

    # Detaillierte Analyse aller Filterkriterien
    print("Analysiere einzelne Filterkriterien...")
    filter_results = df['summary'].apply(analyze_rule_based_filters)

    # Extrahiere die Ergebnisse in separate Spalten
    for criterion in ['future_indicators', 'date_patterns', 'headline_indicators', 'headline_date',
                      'future_indicators_score', 'date_patterns_score', 'headline_indicators_score', 'combined_rule_score']:
        df[f'filter_{criterion}'] = filter_results.apply(
            lambda x: x[criterion])

    # Ergebnisse anzeigen
    announcement_count = df['is_announcement_rule'].sum()
    print(f"Regelbasierte Erkennung: {announcement_count} von {len(df)} Einträgen "
          f"({announcement_count/len(df)*100:.2f}%) sind vermutlich Ankündigungen.")

    # Erstelle und speichere Visualisierungen wenn gewünscht
    if save_visualizations:
        create_rule_based_visualizations(df, output_dir)

    return df


def create_rule_based_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """
    Erstellt Visualisierungen für die regelbasierten Filter.

    Args:
        df: DataFrame mit angewendeten Filterkriterien
        output_dir: Ausgabeverzeichnis für die Visualisierungen
    """
    output_path = Path(output_dir)
    
    # Überprüfe, ob der Ausgabepfad "reports" enthält
    if "reports" not in str(output_path):
        # Wenn nicht, erstelle einen reports/rule_based Unterordner
        parent_dir = output_path.parent if output_path.is_file() else output_path
        output_path = parent_dir / "reports" / "rule_based"
    
    output_path.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(15, 10))

    # 1. Barplot für einzelne Kriterien
    criterion_counts = pd.DataFrame({
        'Kriterium': [
            'Zukunftsindikatoren',
            'Datumsmuster',
            'Überschriften-Indikatoren',
            'Überschrift + Datum'
        ],
        'Anzahl': [
            df['filter_future_indicators'].sum(),
            df['filter_date_patterns'].sum(),
            df['filter_headline_indicators'].sum(),
            df['filter_headline_date'].sum()
        ]
    })

    criterion_counts['Prozent'] = criterion_counts['Anzahl'] / len(df) * 100

    plt.subplot(2, 2, 1)
    sns.barplot(data=criterion_counts, x='Kriterium', y='Prozent')
    plt.xticks(rotation=45, ha='right')
    plt.title('Anteil der gefilterten Einträge pro Kriterium')
    plt.ylabel('Prozent der Einträge')

    # 2. Venn-Diagramm für Überlappungen
    plt.subplot(2, 2, 2)
    try:
        from matplotlib_venn import venn3

        # Erstelle Sets für die drei Hauptkriterien
        future_set = set(df[df['filter_future_indicators']].index)
        date_set = set(df[df['filter_date_patterns']].index)
        headline_set = set(df[df['filter_headline_indicators']].index)

        venn3([future_set, date_set, headline_set],
              ('Zukunftsindikatoren', 'Datumsmuster', 'Überschriften'))
        plt.title('Überlappung der Filterkriterien')
    except ImportError:
        plt.text(0.5, 0.5, "matplotlib-venn nicht installiert",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 3. Gestapeltes Balkendiagramm für kombinierte Effekte
    plt.subplot(2, 2, 3)
    filter_combinations = df[['filter_future_indicators',
                              'filter_date_patterns',
                              'filter_headline_indicators']].sum(axis=1)
    combination_counts = filter_combinations.value_counts().sort_index()
    plt.bar(range(len(combination_counts)), combination_counts)
    plt.xticks(range(len(combination_counts)),
               [f"{i} Kriterien" for i in combination_counts.index])
    plt.title('Anzahl erfüllter Kriterien pro Eintrag')
    plt.ylabel('Anzahl Einträge')

    # 4. Zusammenfassende Statistik
    plt.subplot(2, 2, 4)
    summary_stats = pd.DataFrame({
        'Metrik': [
            'Gesamtanzahl Einträge',
            'Mind. 1 Kriterium erfüllt',
            'Mind. 2 Kriterien erfüllt',
            'Alle Kriterien erfüllt'
        ],
        'Anzahl': [
            len(df),
            (filter_combinations >= 1).sum(),
            (filter_combinations >= 2).sum(),
            (filter_combinations == 3).sum()
        ]
    })
    summary_stats['Prozent'] = summary_stats['Anzahl'] / len(df) * 100

    plt.axis('off')
    plt.table(cellText=summary_stats.values,
              colLabels=summary_stats.columns,
              cellLoc='center',
              loc='center',
              bbox=[0, 0, 1, 1])
    plt.title('Zusammenfassende Statistik')

    plt.tight_layout()
    plt.savefig(output_path / "rule_based_analysis.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"Visualisierungen gespeichert unter: {output_path / 'rule_based_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Regelbasierte Filterung für Gerichtsurteile-Datensatz")
    parser.add_argument("--input", "-i", type=str,
                        required=True, help="Pfad zur Eingabe-CSV-Datei")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="outputs", help="Ausgabeverzeichnis")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Visualisierungen erstellen")
    args = parser.parse_args()

    # Datensatz laden
    df = load_dataset(args.input)

    # Regelbasierte Filter anwenden
    filtered_df = apply_rule_based_filters(
        df, save_visualizations=args.visualize, output_dir=args.output_dir)

    # Ergebnisse speichern
    save_results(filtered_df, args.output_dir, "rule_based_filtered.csv")

    # Bereinigte Version speichern (Optional)
    cleaned_df = filtered_df[~filtered_df['is_announcement_rule']].copy()
    save_results(cleaned_df, args.output_dir, "rule_based_cleaned.csv")

    print(
        f"Bereinigter Datensatz: {len(cleaned_df)} Einträge (von ursprünglich {len(df)})")
    print(
        f"Entfernt: {len(df) - len(cleaned_df)} Einträge ({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()
