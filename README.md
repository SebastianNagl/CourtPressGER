# CourtPressGer

## Hintergrund und Ziel
Wir sind Wissenschaftler der Technischen Universität München im Bereich Legal Technology. Für unsere nächste wissenschaftliche Publikation möchten wir drei Dinge vorstellen: 
1. Eine bereinigte Version eines vorab gescrapten Datensatzes mit etwa 6.5k Einträgen von Gerichtsurteilen mit den dazugehörigen Pressemitteilungen und Metadaten. Die Daten können roh über das dataset.py Skript von HF geladen werden. Für die Bereinigung haben wir in courtpressger/data_cleaning eine Pipeline entwickelt.
2. Synthetische Prompts, mit denen man automatisiert aus den Urteilen Pressemitteilungen generieren kann. 
3. Eine Pipeline, die state of the art Modelle mit den synthetischen Pressemitteilungen queryt und die Ergebnisse neben der tatsächlichen Pressemitteilung speichert.
4. Eine Evaluation der generierten Pressemitteilungen mit Hilfe von menschlichen und automatisierten Metriken.
Neben den Pipeline-Skripten haben wir für unsere Analysen auch immer ein Jupyter Notebook für die Arbeitsschritte; abrufbar unter notebooks/.

## Aktuelle Aufgaben und Probleme
Aktuelle Aufgaben können work in progress sein; immer erst mal kontrollieren, dann lösen. Sobald was davon erledigt ist, bitte [erledigt] zu Beginn der Aufgabe schreiben; ich kontrolliere dann bei Gelegenheit. 


# Struktur
Das Projekt folgt im Kern der Cookiecutter Data Science Projektstruktur. Skripte und Module sind unter courtpressger/ angeordnet.

## Projektübersicht
Das CourtPressGER-Projekt ist in mehrere Module unterteilt, die verschiedene Funktionalitäten bereitstellen:

1. **Datenbereinigung**: Module zur Reinigung und Vorbereitung der Rohdaten.
2. **Synthetische Prompts**: Module zur synthetisierung von Prompts für die Erzeugung von Pressemitteilungen aus Gerichtsurteilen.
3. **Generierung**: Module zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs.
4. **Evaluierung**: Module zur Bewertung der generierten Pressemitteilungen.

## Projektstruktur
```
CourtPressGER/
├── courtpressger/               # Hauptmodul der Anwendung; enthält alle Module und Skripte
│   ├── data_cleaning/           # Module zur Datenbereinigung
│   │   ├── cli.py               # Kommandozeilenschnittstelle für die Datenbereinigung
│   │   ├── rule_based.py        # Regelbasierte Bereinigungsmethoden
│   │   └── utils.py             # Hilfsfunktionen
│   ├── synthetic_prompts/       # Module für synthetische Prompts
│   │   ├── cli.py               # Kommandozeilenschnittstelle für Prompt-Generierung
│   │   ├── generator.py         # Generator für synthetische Prompts
│   │   ├── rate_limiter.py      # API-Ratenbegrenzung
│   │   ├── sanitizer.py         # Bereinigung und Formatierung von Prompts
│   │   └── verify_checkpoints.py # Validierung von Checkpoints
│   ├── generation/              # Module zur Generierung von Pressemitteilungen
│   │   ├── cli.py               # Kommandozeilenschnittstelle für die Generierung
│   │   └── pipeline.py          # Pipeline zur Generierung von Pressemitteilungen
│   ├── evaluation/              # Module zur Evaluierung
│   │   ├── cli.py               # Kommandozeilenschnittstelle für die Evaluierung
│   │   ├── metrics.py           # Metriken für die Evaluierung
│   │   ├── models.py            # Implementierung verschiedener LLMs
│   │   ├── pipeline.py          # Pipeline zur Evaluierung der Modelle
│   │   └── utils.py             # Hilfsfunktionen für Visualisierung und Analyse
│   ├── __init__.py              # Initialisierung des Pakets
│   ├── dataset.py               # Datenverwaltung und -zugriff
│   ├── download_eurobert.py     # Skript zum Herunterladen des EuroBERT-Modells
│   ├── download_eurollm.py      # Skript zum Herunterladen des EuroLLM-9B-Modells
│   ├── download_llama3_8b.py    # Skript zum Herunterladen des Llama-3.1-8B-Modells
│   ├── download_teuken.py       # Skript zum Herunterladen des Teuken-7B-Modells
│   └── main.py                  # Haupteinstiegspunkt der Anwendung
├── data/                        # Datendateien
│   ├── raw/                     # Rohdaten
│   ├── interim/                 # Zwischendaten
│   ├── processed/               # Verarbeitete Daten
│   ├── checkpoints/             # Checkpoints für lange Verarbeitungsprozesse
│   ├── generation/              # Generierte Pressemitteilungen
│   └── evaluation/              # Evaluierungsergebnisse
├── notebooks/                   # Jupyter Notebooks für Analysen
├── tests/                       # Testmodule
│   ├── evaluation/              # Tests für Evaluierungskomponenten
│   │   ├── test_pipeline.py     # Tests für die Evaluierungspipeline
│   │   └── test_factual_metrics.py # Tests für QAGS und FactCC Metriken
│   ├── examples/                # Beispieldaten für Tests
│   ├── test_generation_pipeline.py # Tests für die Generierungspipeline
│   ├── test_teuken_model.py     # Tests für das Teuken-Modell
│   ├── test_checkpoint_generation.py # Tests für die Checkpoint-Funktionalität
│   └── test_csv_validator.py    # Tests für die CSV-Validierung
├── models/                      # Modellierte Daten und Modellkonfigurationen
│   ├── teuken/                  # Teuken-7B-Modell für die lokale Generierung
│   ├── eurollm/                 # EuroLLM-9B-Modell für die lokale Generierung
│   ├── eurobert/                # EuroBERT-Modell für BERTScore
│   ├── generation_config.json   # Konfiguration für die Generierungsmodelle
│   ├── evaluation_config.json   # Konfiguration für die Evaluierungsmodelle
│   └── evaluation_models_config.json # Erweiterte Konfiguration für Evaluierungsmodelle
├── references/                  # Referenzdokumente und Schemadefitionen
├── reports/                     # Generierte Berichte und Visualisierungen
├── pyproject.toml               # Projekteinstellungen und Abhängigkeiten
├── uv.lock                      # Lock-Datei für uv-Paketmanager
├── .env                         # Umgebungsvariablen und API-Keys
├── Makefile                     # Automatisierungsscripts für häufige Aufgaben
└── README.md                    # Projektdokumentation
```

## Funktionalitäten
Das Projekt bietet folgende Hauptfunktionalitäten:

### Datenbereinigung
- Bereinigung von Rohdaten mit Gerichtsurteilen und Pressemitteilungen
- Regelbasierte Filterung und Normalisierung
- Semantische Ähnlichkeitsanalyse zur Validierung der Zuordnungen

### Synthetische Prompts
- Generierung von synthetischen Prompts für LLMs
- Verschiedene Prompt-Strategien zur Pressemitteilungsgenerierung
- Funktionen für API-Ratenbegrenzung und Checkpoint-Verwaltung
- Validierung und Bereinigung von generierten Prompts

### Generierung von Pressemitteilungen
- Pipeline zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs
- Unterstützung für verschiedene Modelltypen:
  - OpenAI Modelle (GPT-4o)
  - DeepInfra Modelle (Llama-3-70B)
  - Lokale Modelle (Teuken-7B, EuroLLM-9B, Llama-3-8B)
- Modellkategorien:
  - Große Modelle: GPT-4o, Llama-3-70B
  - Kleine Modelle: Teuken-7B, Llama-3-8B, EuroLLM-9B
- Checkpoint-System zur Fortsetzung unterbrochener Generierungen
- Speicherung der generierten Pressemitteilungen in verschiedenen Formaten (JSON, CSV)

### Evaluierung
- Pipeline zur Evaluierung verschiedener LLMs
- Unterstützung für OpenAI, Hugging Face und lokale Modelle
- Berechnung verschiedener Metriken zur Textähnlichkeit und sachlichen Konsistenz:
  - ROUGE (Rouge-1, Rouge-2, Rouge-L)
  - BLEU (BLEU-1 bis BLEU-4)
  - METEOR
  - BERTScore (verwendet EuroBERT-Modell)
  - QAGS (Question Answering for evaluating Generated Summaries)
  - FactCC (Evaluierung der faktischen Konsistenz)
  - LLM-as-a-Judge (Bewertung durch Claude 3.7 Sonnet)
- Visualisierung und Reporting-Funktionen für Ergebnisanalyse
- Checkpoint-System für langläufige Evaluierungen

#### Sachliche Konsistenzmetriken
Seit neuestem unterstützt das Projekt auch fortgeschrittene Metriken zur Bewertung der sachlichen Konsistenz zwischen Gerichtsurteilen und generierten Pressemitteilungen:

- **QAGS (Question Answering for evaluating Generated Summaries)**
  - Generiert Fragen aus den Pressemitteilungen
  - Beantwortet diese Fragen mit den Gerichtsurteilen als Kontext
  - Vergleicht die Antworten, um zu prüfen, ob die Pressemitteilung sachlich korrekt ist
  
- **FactCC (Factual Consistency Check)**
  - Extrahiert Behauptungen aus den Pressemitteilungen
  - Überprüft jede Behauptung auf Konsistenz mit dem Gerichtsurteil
  - Berechnet einen Gesamtscore für die faktische Konsistenz

- **LLM-as-a-Judge (Claude 3.7 Sonnet)**
  - Verwendet Claude 3.7 Sonnet zur Bewertung der generierten Pressemitteilungen
  - Bewertet anhand verschiedener Kriterien (faktische Korrektheit, Vollständigkeit, Klarheit, Struktur)
  - Vergleicht die generierte Pressemitteilung optional mit der Referenzpressemitteilung
  - Liefert sowohl numerische Bewertungen (1-10) als auch detaillierte Begründungen
  - Berechnet einen Gesamtscore über alle Bewertungskriterien

Diese Metriken können über folgende Befehle ausgeführt werden:
- `make eval-factual`: Aktiviert QAGS und FactCC für die sachliche Konsistenzprüfung
- `make eval-llm-judge`: Aktiviert die Bewertung durch Claude 3.7 Sonnet
- `make eval-full`: Führt alle Evaluierungsmetriken inklusive sachlicher Konsistenzprüfung und LLM-as-a-Judge aus

Alternativ können diese Funktionen über entsprechende Kommandozeilenoptionen aktiviert werden:
- `--enable-factual-consistency`: Aktiviert QAGS und FactCC
- `--enable-llm-as-judge`: Aktiviert die Bewertung durch Claude 3.7 Sonnet

## Package Management
Das Projekt nutzt uv, um Pakete und Venvs zu verwalten. Im besten Fall sollen pakete durch uv add hinzugefügt werden, nur im Ausnahmefall durch uv pip install. Es wird eine einzige virtuelle Umgebung unter .venv verwendet, die für alle Aufgaben (CPU und GPU) geeignet ist.

## Lizenz
Das Projekt ist lizenziert unter der MIT-Lizenz.
