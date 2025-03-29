"""
Regelbasierte Filterung für die Datensatzbereinigung.

Identifiziert Ankündigungen und nicht-urteilsbezogene Mitteilungen
anhand vordefinierter Regeln und Keyword-Mustern.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import argparse
from collections import Counter
from typing import Tuple, List, Dict, Any, Optional

from courtpressger.data_cleaning.utils import load_dataset, save_results


def is_announcement_rule_based(row) -> Tuple[bool, Optional[str]]:
    """
    Erkennt Ankündigungen und nicht urteilsbezogene Mitteilungen anhand von Keywords und Mustern.
    
    Args:
        row: Zeile des DataFrames mit den zu prüfenden Daten
    
    Returns:
        Tuple mit (is_announcement, matching_criteria_str)
    """
    if pd.isna(row['summary']):
        return False, None
    
    summary = str(row['summary']).lower()
    
    # 1. Keywords, die auf Ankündigungen hindeuten
    future_indicators = [
        'ankündigung', 'terminankündigung', 'terminhinweis', 'hinweis auf termin',
        'wird verhandelt', 'wird verhandeln', 'wird.*stattfinden', 'findet statt',
        'einladung', 'pressetermin', 'pressekonferenz', 'veranstaltung',
        'wochenvorschau', 'jahrespressekonferenz', 'pressegespräch', 'rundgang',
        'beginnt am', 'laden ein', 'lädt ein', 'sitzung vom', 
    ]
    
    # 2. Regex-Muster für Datumsangaben in der Zukunft (relativ zum Erstelldatum)
    date_patterns = [
        r'am\s+(\d{1,2})\.(\d{1,2})\.(\d{4})',  # "am 15.05.2023"
        r'am\s+(\d{1,2})\.(\d{1,2})\.',         # "am 15.05."
        r'vom\s+(\d{1,2})\.(\d{1,2})\.(\d{4})?', # "vom 15.05.2023" oder "vom 15.05."
        r'terminiert auf den (\d{1,2})\.(\d{1,2})\.(\d{4})?',  # "terminiert auf den 15.05.2023"
        r'findet\s+am\s+(\d{1,2})\.(\d{1,2})\.(\d{4})?\s+statt', # "findet am 15.05.2023 statt"
        r'verhandelt\s+am\s+(\d{1,2})\.(\d{1,2})\.(\d{4})?', # "verhandelt am 15.05.2023"
    ]
    
    # 3. Überschriften/Anfänge, die auf Ankündigungen hindeuten
    headline_indicators = [
        'presseinformation', 'information für die presse',
        'medieninformation', 'zur information', 
        'terminvorschau', 'jahresbericht', 'geschäftsbericht', 'tätigkeitsbericht',
        'stellenausschreibung', 'personelle veränderungen', 'neuer präsident', 
        'akkreditierung','sehr geehrt'
    ]
    
    # 4. Prüfen auf Keywords am Anfang des Textes (größeres Gewicht)
    summary_start = summary[:500]
    
    # Speichere die gefundenen Kriterien
    matching_criteria = []
    
    # Headline Prüfung mit genauen Matches
    for indicator in headline_indicators:
        if indicator in summary_start:
            matching_criteria.append(f"Headline: {indicator}")
    
    # Keyword Prüfung
    for indicator in future_indicators:
        if re.search(r'\b' + re.escape(indicator) + r'\b', summary, re.IGNORECASE):
            matching_criteria.append(f"Keyword: {indicator}")
    
    # Prüfen, ob die Pressemitteilung ein Datum enthält
    # Wir extrahieren hier das Datum und vergleichen es mit dem Veröffentlichungsdatum
    future_date_found = False
    current_date = None
    
    # Versuchen, das Veröffentlichungsdatum zu parsen
    try:
        if 'date' in row and pd.notna(row['date']):
            current_date = pd.to_datetime(row['date'], errors='coerce')
    except:
        current_date = None
    
    # Wenn kein Veröffentlichungsdatum vorhanden ist, können wir keine zeitliche Prüfung machen
    if current_date is not None:
        # Aktuelle Jahreszahl für unvollständige Datumsmuster
        current_year = current_date.year
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, summary, re.IGNORECASE):
                try:
                    groups = match.groups()
                    if len(groups) >= 2:
                        day = int(groups[0])
                        month = int(groups[1])
                        # Jahr ist nicht immer angegeben, daher nehmen wir das aktuelle Jahr, wenn es fehlt
                        year = int(groups[2]) if len(groups) > 2 and groups[2] is not None else current_year
                        
                        # Datumsvalidierung (einfache Prüfung)
                        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                            # Datum aus dem Text konstruieren
                            event_date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors='coerce')
                            
                            # Prüfen, ob das Datum gültig ist und in der Zukunft liegt
                            if pd.notna(event_date) and event_date > current_date:
                                # Prüfen, ob das Datum mindestens einen Tag später ist
                                days_diff = (event_date - current_date).days
                                if days_diff >= 1:
                                    future_date_found = True
                                    matching_criteria.append(f"Datum: {match.group(0)} (+{days_diff} Tage)")
                except Exception as e:
                    # Bei Fehlern in der Datumsverarbeitung ignorieren wir diesen Match
                    continue
    
    # Unspezifische zukünftige Zeitangaben (als fallback)
    future_time_indicators = [
        r'in der kommenden woche',
        r'in der nächsten woche',
        r'demnächst'
    ]
    
    for pattern in future_time_indicators:
        if re.search(pattern, summary, re.IGNORECASE):
            matching_criteria.append(f"Zeitraum: {pattern}")
            # Diese gelten immer als zukünftig
            future_date_found = True
    
    # Kombinierte Entscheidung mit unterschiedlicher Gewichtung
    headline_match = any("Headline:" in criterion for criterion in matching_criteria)
    keyword_match = any("Keyword:" in criterion for criterion in matching_criteria)
    
    # Eine Ankündigung muss entweder einen Headline-Indikator haben,
    # oder eine Kombination aus Keyword und Datums-Hinweis
    if headline_match or (keyword_match and future_date_found):
        return True, " | ".join(matching_criteria)
    
    return False, None  # Default: keine Ankündigung


def apply_rule_based_filters(
    df: pd.DataFrame,
    save_visualizations: bool = False,
    output_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Wendet regelbasierte Filter auf den Datensatz an.
    
    Args:
        df: DataFrame mit Gerichtsurteilen und Pressemitteilungen
        save_visualizations: Ob Visualisierungen gespeichert werden sollen
        output_dir: Ausgabeverzeichnis für Visualisierungen
        
    Returns:
        DataFrame mit angefügten Filterungsergebnissen
    """
    print("\n## Regelbasierte Filterung")
    
    # Anwenden der regelbasierten Methode auf den Datensatz
    results = df.apply(is_announcement_rule_based, axis=1)
    df['is_announcement_rule'] = [result[0] for result in results]
    df['matching_criteria'] = [result[1] for result in results]
    
    # Auswertung der regelbasierten Methode
    announcement_count = df['is_announcement_rule'].sum()
    print(f"Regelbasierte Erkennung: {announcement_count} von {len(df)} Einträgen " +
          f"({announcement_count/len(df)*100:.2f}%) sind vermutlich Ankündigungen.")
    
    # Erstelle Visualisierungen, wenn gewünscht
    if save_visualizations:
        create_rule_based_visualizations(df, output_dir)
    
    return df


def create_rule_based_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """
    Erstellt Visualisierungen für die regelbasierte Filterung.
    
    Args:
        df: DataFrame mit regelbasierten Filterungsergebnissen
        output_dir: Ausgabeverzeichnis für Visualisierungen
    """
    output_path = Path(output_dir)
    
    # Überprüfe, ob der Ausgabepfad "reports" enthält
    if "reports" not in str(output_path):
        # Wenn nicht, erstelle einen reports/rule_based Unterordner
        parent_dir = output_path.parent if output_path.is_file() else output_path
        output_path = parent_dir / "reports" / "rule_based"
    
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Extrahiere alle Kriterien für die Analyse
    all_criteria = []
    announcement_examples = df[df['is_announcement_rule']]
    
    if not announcement_examples.empty:
        for criteria_str in announcement_examples['matching_criteria'].dropna():
            all_criteria.extend(criteria_str.split(" | "))
        
        # 1. Visualisierung der Kriterien-Verteilung
        criteria_types = ['Headline', 'Keyword', 'Datum']
        type_counts = {t: sum(1 for c in all_criteria if c.startswith(f"{t}:")) for t in criteria_types}
        
        plt.figure(figsize=(10, 6))
        plt.bar(type_counts.keys(), type_counts.values())
        plt.title('Verteilung der Kriterien-Typen')
        plt.ylabel('Anzahl')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(output_path / "criteria_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Visualisierung der häufigsten Kriterien
        criteria_counts = Counter(all_criteria)
        
        plt.figure(figsize=(12, 8))
        top_criteria = dict(criteria_counts.most_common(15))
        plt.barh(list(reversed(list(top_criteria.keys()))), 
                 list(reversed(list(top_criteria.values()))))
        plt.title('Top 15 Häufigste Match-Kriterien')
        plt.xlabel('Anzahl')
        plt.tight_layout()
        plt.savefig(output_path / "top_criteria.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Verteilung der Ankündigungen nach Gericht (falls verfügbar)
        if 'court' in df.columns:
            plt.figure(figsize=(10, 6))
            court_announcement_counts = df[df['is_announcement_rule']].groupby('court').size()
            court_total_counts = df.groupby('court').size()
            court_announcement_percentage = (court_announcement_counts / court_total_counts * 100).sort_values(ascending=False)
            
            court_announcement_percentage.plot(kind='bar')
            plt.title('Anteil der Ankündigungen nach Gericht')
            plt.xlabel('Gericht')
            plt.ylabel('Prozentsatz (%)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_path / "court_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualisierungen gespeichert unter: {output_path}")


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
    
    # Regelbasierte Filterung anwenden
    df_filtered = apply_rule_based_filters(
        df,
        save_visualizations=args.visualize,
        output_dir=args.output_dir
    )
    
    # Ergebnisse speichern
    save_results(df_filtered, args.output_dir, "rule_based_results.csv")
    
    # Bereinigte Version speichern (optional)
    cleaned_df = df_filtered[~df_filtered['is_announcement_rule']].copy()
    removed_df = df_filtered[df_filtered['is_announcement_rule']].copy()
    
    # Status-Spalte hinzufügen für das neue Format
    cleaned_df['status'] = 'cleaned'
    removed_df['status'] = 'removed'
    
    # Speichern als getrennte Dateien
    interim_dir = Path(args.output_dir) / "interim"
    interim_dir.mkdir(exist_ok=True, parents=True)
    
    cleaned_path = interim_dir / "cleaned.csv"
    removed_path = interim_dir / "removed.csv"
    
    cleaned_df.to_csv(cleaned_path, index=False)
    removed_df.to_csv(removed_path, index=False)
    
    print(f"\nBereinigte Daten gespeichert: {len(cleaned_df)} Einträge in {cleaned_path}")
    print(f"Herausgefilterte Ankündigungen gespeichert: {len(removed_df)} Einträge in {removed_path}")
    print(f"\nZusammenfassung:")
    print(f"- Ursprüngliche Datensatzgröße: {len(df)} Einträge")
    print(f"- Entfernte Ankündigungen: {len(removed_df)} Einträge ({len(removed_df)/len(df)*100:.2f}%)")
    print(f"- Verbleibende Daten: {len(cleaned_df)} Einträge ({len(cleaned_df)/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main() 