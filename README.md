# CourtPressGER

Sammlung und Verarbeitung von Gerichtsurteilen und zugehörigen Pressemitteilungen.

## Installation
#todo Ich mag die Lösung über uv pip install nicht; würde das gern noch umziehen auf pures uv add.

```
uv pip install -e .               # Reguläre Installation
uv pip install -e ".[gpu]"        # Mit GPU-optimierten Abhängigkeiten
```

## Verwendung

### Datensatz laden

```python
from courtpressger.dataset import load_dataset

# Laden des kompletten Datensatzes
dataset = load_dataset()

# Laden eines spezifischen Untergruppe
dataset = load_dataset(court="BGH")  # Nur Entscheidungen des BGH
```

### Bereinigung des Datensatzes

Die Module zur Bereinigung des Datensatzes ermöglichen die Identifizierung und Entfernung von Pressemitteilungen, die keinen direkten Bezug zu einem ergangenen Urteil haben (z.B. Ankündigungen zukünftiger Verhandlungen).

#### 1. Verwendung über die Kommandozeile

Für rechenintensive Verarbeitungen wird empfohlen, die Skripte direkt auf einem GPU-Server auszuführen:

```bash
# Installation mit GPU-Unterstützung
uv pip install -e ".[gpu]"

# Kombinierte Bereinigungspipeline ausführen
courtpressger-clean combined --input data/german_courts.csv --output_dir outputs --visualize

# Oder einzelne Methoden
courtpressger-clean rule_based --input data/german_courts.csv --output_dir outputs --visualize
courtpressger-clean similarity --input data/german_courts.csv --output_dir outputs --visualize
courtpressger-clean ml --input data/german_courts.csv --output_dir outputs --visualize
courtpressger-clean clustering --input data/german_courts.csv --output_dir outputs --visualize
```

Alle generierten Visualisierungen werden im Unterordner `reports` des angegebenen Ausgabeverzeichnisses gespeichert, um eine übersichtliche Strukturierung zu gewährleisten.

#### 2. Verwendung im Code

```python
from courtpressger.data_cleaning.utils import load_dataset
from courtpressger.data_cleaning.combined_pipeline import run_combined_pipeline

# Datensatz laden
df = load_dataset("data/german_courts.csv")

# Kombinierte Bereinigungspipeline ausführen
cleaned_df = run_combined_pipeline(
    df,
    min_votes=2,
    save_visualizations=True,
    output_dir="outputs"
)

# Bereinigten Datensatz speichern
cleaned_df[~cleaned_df['is_irrelevant_combined']].to_csv("data/cleaned_courts.csv", index=False)
```

Die Visualisierungen sind anschließend im Ordner `outputs/reports` verfügbar.

#### 3. Bereinigungsmethoden

Das Paket enthält folgende Bereinigungsansätze:

1. **Regelbasierte Filter**: Erkennung von Ankündigungen mit Keywords und Mustern
2. **Semantische Ähnlichkeit**: Berechnung der Kosinus-Ähnlichkeit zwischen Urteilen und Pressemitteilungen
3. **Überwachtes Machine Learning**: Traininierter Klassifikator zur Identifikation irrelevanter Mitteilungen
4. **Unüberwachtes Clustering**: Gruppierung ähnlicher Pressemitteilungen zur Identifikation von Mustern
5. **Kombinierter Ansatz**: Mehrheitsentscheidung aller Methoden für robuste Ergebnisse

Für Best Performance wird die Verwendung eines GPU-Servers mit CUDA 12.x und den [RAPIDS](https://rapids.ai)-Bibliotheken empfohlen.

## Projekt-Struktur

```
.
├── courtpressger/               # Hauptpaket
│   ├── __init__.py              # Package-Initialisierung
│   ├── dataset.py               # Funktionen zum Laden des Datensatzes
│   ├── main.py                  # CLI-Funktionalität
│   └── data_cleaning/           # Module zur Datenbereinigung
│       ├── __init__.py
│       ├── utils.py             # Hilfsfunktionen
│       ├── rule_based.py        # Regelbasierte Filter
│       ├── semantic_similarity.py # Semantische Ähnlichkeitsanalyse
│       ├── ml_classifier.py     # ML-Klassifikation 
│       ├── clustering.py        # Clustering-Methoden
│       ├── combined_pipeline.py # Kombinierte Pipeline
│       └── cli.py               # CLI für Bereinigungsmodule
├── data/                        # Datenverzeichnis
├── notebooks/                   # Jupyter Notebooks
│   └── bereinigung_notebook.ipynb # Notebook zur Datenbereinigung
├── reports/                     # Visualisierungen und Berichte
├── tests/                       # Testverzeichnis
├── LICENSE
├── README.md
├── pyproject.toml               # Projektmetadaten und Abhängigkeiten
└── Makefile                     # Build-Skripte
```

## Datenquellen

Grundlage sind Gerichtsurteile und zugehörige Pressemitteilungen der deutschen Obergerichte:

- Bundesgerichtshof (BGH)
- Bundesverfassungsgericht (BVerfG)
- Bundesverwaltungsgericht (BVerwG)
- Bundesfinanzhof (BFH)
- Bundesarbeitsgericht (BAG)
- Bundessozialgericht (BSG)

## Lizenz

[MIT](LICENSE)