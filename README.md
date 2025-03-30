# CourtPressGer

## Hintergrund und Ziel
Wir sind Wissenschaftler der Technischen Universität München im Bereich Legal Technology. Für unsere nächste wissenschaftliche Publikation möchten wir drei Dinge vorstellen: 
1. Eine bereinigte Version eines vorab gescrapten Datensatzes mit etwa 6.5k Einträgen von Gerichtsurteilen mit den dazugehörigen Pressemitteilungen und Metadaten.
2. Synthetische Prompts, mit denen man automatisiert aus den Urteilen Pressemitteilungen generieren kann.
3. Eine Evaluation der generierten Pressemitteilungen mit Hilfe von menschlichen und automatisierten Metriken.

## Aktuelle Aufgaben und Probleme
Aktuelle Aufgaben können work in progress sein; immer erst mal kontrollieren, dann lösen. Sobald was davon erledigt ist, bitte [erledigt] zu Beginn der Aufgabe schreiben; ich kontrolliere dann bei Gelegenheit.

1. [erledigt] Es gab ein kleines Git Problem (ich); teilweise konnten Inhalte wiederverwendet werden. Ggf sind Verweise auf Inhalte vorhanden, die nicht mehr bestehen, und unnötige Pakete werden geladen. Das müsste mal aufgeräumt werden.
2. [erledigt] Um zu verstehen, was vorhanden ist, hätte ich gern erst mal eine Übersicht über das Projekt und die Funktionalitäten hier im README.

# Struktur

## Struktur des Projekts
Das Projekt ist wie folgt strukturiert:

```
CourtPressGER/
├── courtpressger/               # Hauptmodulordner
│   ├── __init__.py              # Initialisierungsdatei
│   ├── main.py                  # Haupteinstiegspunkt
│   ├── dataset.py               # Datensatzverarbeitung
│   ├── data_cleaning/           # Module zur Datenbereinigung
│   │   ├── __init__.py
│   │   ├── cli.py               # Kommandozeilenschnittstelle
│   │   ├── rule_based.py        # Regelbasierte Reinigung
│   │   ├── semantic_similarity.py # Semantische Ähnlichkeitsanalyse
│   │   └── utils.py             # Hilfsfunktionen
│   └── synthetic_prompts/       # Module für synthetische Prompts
│       ├── __init__.py
│       ├── cli.py               # Kommandozeilenschnittstelle
│       ├── generator.py         # Prompt-Generator
│       ├── rate_limiter.py      # API-Ratenbegrenzung
│       ├── sanitizer.py         # Datenbereinigung
│       └── verify_checkpoints.py # Checkpoint-Validierung
├── data/                        # Datenordner
│   ├── raw/                     # Rohdaten
│   ├── interim/                 # Zwischendaten
│   ├── processed/               # Verarbeitete Daten
│   └── checkpoints/             # Verarbeitungscheckpoints
├── models/                      # Modellordner
├── notebooks/                   # Jupyter Notebooks
├── references/                  # Verweise und Literaturreferenzen
├── reports/                     # Berichte und Visualisierungen
├── tests/                       # Testordner
├── .env                         # Umgebungsvariablen (API-Schlüssel)
├── .gitignore                   # Git-Ignorierungsmuster
├── Makefile                     # Build-Automatisierung
├── pyproject.toml               # Projektmetadaten und Abhängigkeiten
└── pytest.ini                   # Pytest-Konfiguration
```

## Funktionalitäten
Das Projekt bietet folgende Hauptfunktionalitäten:

1. **Daten-Pipeline**: 
   - Laden und Vorverarbeiten des deutschen Gerichtsurteil-Datensatzes
   - Bereinigung und Filterung von Daten
   - Speicherung in verschiedenen Verarbeitungsstufen (raw, interim, processed)

2. **Datenbereinigung**:
   - Regelbasierte Reinigung von Texten
   - Semantische Ähnlichkeitsanalyse zwischen Urteilen und Pressemitteilungen
   - Spezielle Sanitierung für juristische Texte

3. **Synthetische Prompt-Generierung**:
   - Generierung von Prompts aus Gerichtsurteilen
   - Integration mit externen API-Diensten (Anthropic)
   - Ratenbegrenzung für API-Aufrufe

4. **Kommandozeilentools**:
   - `courtpressger`: Haupteinstiegspunkt
   - `courtpressger-clean`: Datenbereinigungstool
   - `courtpressger-prompt`: Tool zur Prompt-Generierung

## Package Management
Das Projekt nutzt uv, um Pakete und Venv zu verwalten. Im besten Fall sollen pakete durch uv add hinzugefügt werden, nur im Ausnahmefall durch uv pip install.

## Daten
Die Daten liegen im `data` Ordner und in verschiedenen Subordnern. Abhängig vom Bearbeitungsstand gibt es raw, intermediate und processed Daten. Daneben gibt es `checkpoints`, in denen die Zwischenergebnisse der automatisierten Verarbeitung gespeichert werden.

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz.