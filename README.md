# CourtPressGER

Sammlung und Verarbeitung von Gerichtsurteilen und zugehörigen Pressemitteilungen.

## Installation

Das Projekt unterstützt zwei verschiedene Installationen, abhängig von Ihrem System:

### CPU-Installation (für MacOS und Systeme ohne CUDA)

```bash
# Erstelle eine CPU-spezifische virtuelle Umgebung
make venv-cpu

# Aktiviere die Umgebung
source .venv-cpu/bin/activate
```

### GPU-Installation (für Linux-Systeme mit CUDA 11.8)

```bash
# Erstelle eine GPU-spezifische virtuelle Umgebung
make venv-gpu

# Aktiviere die Umgebung
source .venv-gpu/bin/activate
```

### Systemanforderungen

- **CPU-Version**:
  - Python 3.12 oder höher
  - MacOS oder Linux ohne CUDA
  - Mindestens 16GB RAM empfohlen

- **GPU-Version**:
  - Python 3.12 oder höher
  - Linux mit NVIDIA GPU
  - CUDA 11.8 Toolkit
  - Mindestens 16GB RAM empfohlen
  - Mindestens 8GB GPU-Speicher empfohlen

### Entwicklungsumgebung einrichten

Für Entwickler, die am Projekt mitarbeiten möchten:

```bash
# Virtuelle Umgebung erstellen (CPU oder GPU)
make venv-cpu  # oder make venv-gpu

# Abhängigkeiten synchronisieren
make sync

# Code formatieren
make format

# Tests ausführen
make test
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

### Generierung von synthetischen Prompts

Das Projekt ermöglicht die Generierung von synthetischen Prompts auf Basis von Gerichtsurteilen und zugehörigen Pressemitteilungen mittels der Anthropic Claude API.

#### 1. Verwendung über die Kommandozeile

```bash
# Synthetische Prompts generieren
courtpressger-prompts --input data/cleaned.csv --output data/processed/court_press_with_synthetic_prompts_final.csv --checkpoint-dir checkpoints --batch-size 5

# Test-Modus für einen einzelnen Eintrag
courtpressger-prompts --input data/cleaned.csv --output test_output.csv --test-single --test-idx 10
```

Wichtige Parameter:

- `--input, -i`: Pfad zur Eingabe-CSV mit Gerichtsurteilen (`judgement`) und Pressemitteilungen (`summary`)
- `--output, -o`: Pfad für die Ausgabe-CSV-Datei mit hinzugefügten synthetischen Prompts
- `--checkpoint-dir, -c`: Verzeichnis für Checkpoints zur Fortsetzung bei Unterbrechungen
- `--model, -m`: Zu verwendendes Claude-Modell (Standard: `claude-3-7-sonnet-20250219`)
- `--batch-size, -b`: Anzahl der Elemente pro Batch (Standard: 5)
- `--start-idx, -s`: Startindex für die Verarbeitung (Standard: 0)
- `--save-interval`: Speicherintervall für Checkpoints (Standard: 5)
- `--fix-errors`: Fehlerhafte Einträge erneut verarbeiten
- `--api-key`: Anthropic API-Schlüssel (alternativ über ANTHROPIC_API_KEY Umgebungsvariable)
- `--env-file`: Pfad zur .env-Datei mit ANTHROPIC_API_KEY

#### 2. Verwendung im Code

```python
from courtpressger.synthetic_prompts.generator import generate_synthetic_prompt, process_batch
import pandas as pd
import anthropic

# Initialisiere Client
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# Lade Datensatz
df = pd.read_csv("data/cleaned.csv")

# Einzelnen Prompt generieren
synthetic_prompt = generate_synthetic_prompt(
    court_ruling=df.iloc[0]['judgement'],
    press_release=df.iloc[0]['summary'],
    client=client
)

# Batch-Verarbeitung
results_df = process_batch(
    df,
    batch_size=5,
    start_idx=0,
    save_interval=5,
    checkpoint_dir="checkpoints",
    output_prefix="synthetic_prompts",
    client=client
)
```

## Projekt-Struktur

```
.
├── courtpressger/                # Hauptpaket
│   ├── __init__.py               # Package-Initialisierung
│   ├── dataset.py                # Funktionen zum Laden des Datensatzes
│   ├── main.py                   # CLI-Funktionalität
│   ├── data_cleaning/            # Module zur Datenbereinigung
│   │   ├── __init__.py
│   │   ├── cli.py                # CLI für Bereinigungsmodule
│   │   ├── utils.py              # Hilfsfunktionen
│   │   ├── rule_based.py         # Regelbasierte Filter
│   │   ├── semantic_similarity.py # Semantische Ähnlichkeitsanalyse
│   │   ├── ml_classifier.py      # ML-Klassifikation 
│   │   ├── clustering.py         # Clustering-Methoden
│   │   └── combined_pipeline.py  # Kombinierte Pipeline
│   └── synthetic_prompts/        # Module zur Generierung synthetischer Prompts
│       ├── __init__.py
│       ├── cli.py                # CLI für Prompt-Generierung
│       ├── generator.py          # Kernfunktionalität zur Generierung
│       └── rate_limiter.py       # Ratenbegrenzung für API-Anfragen
├── data/                         # Datenverzeichnis
│   ├── interim/                  # Zwischenergebnisse
│   │   ├── cleaned.csv           # Bereinigter Datensatz
│   │   └── removed.csv           # Entfernte Einträge
│   ├── processed/                # Aufbereitete Datensätze
│   │   ├── court_press_with_synthetic_prompts_essential.csv
│   │   └── court_press_with_synthetic_prompts_final.csv
│   └── raw/                      # Rohdaten
│       └── german_courts.csv     # Ausgangsdatensatz
├── models/                       # Gespeicherte Modelle
├── notebooks/                    # Jupyter Notebooks
│   ├── bereinigung.ipynb         # Notebook zur Datenbereinigung
│   ├── deskriptiv.ipynb          # Deskriptive Analysen
│   ├── synthetic_prompts.ipynb   # Notebook zur Prompt-Generierung
│   └── checkpoints/              # Notebook-Checkpoints
├── checkpoints/                  # Checkpoints für Prompt-Generierung
├── reports/                      # Visualisierungen und Berichte
│   └── data_cleaning/            # Berichte zur Datenbereinigung
├── tests/                        # Testverzeichnis
├── LICENSE
├── README.md
├── pyproject.toml                # Projektmetadaten und Abhängigkeiten
├── uv.lock                       # Dependency lock file
└── Makefile                      # Build-Skripte
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